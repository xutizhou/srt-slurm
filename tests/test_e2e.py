# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests with mocked SLURM environment using real recipe files."""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Path to real recipe files
RECIPES_DIR = Path(__file__).parent.parent / "recipies"


def get_recipe_files() -> list[Path]:
    """Get all recipe YAML files."""
    if not RECIPES_DIR.exists():
        return []
    return list(RECIPES_DIR.rglob("*.yaml"))


@pytest.fixture
def mock_slurm_env():
    """Mock SLURM environment variables."""
    env = {
        "SLURM_JOB_ID": "12345",
        "SLURM_JOBID": "12345",
        "SLURM_NODELIST": "node[01-10]",
        "SLURM_JOB_NUM_NODES": "10",
        "SRTCTL_SOURCE_DIR": str(Path(__file__).parent.parent),
    }
    with patch.dict(os.environ, env, clear=False):
        yield env


@pytest.fixture
def mock_scontrol():
    """Mock scontrol show hostnames."""

    def mock_run(cmd, **kwargs):
        if cmd[0] == "scontrol" and "hostnames" in cmd:
            result = MagicMock()
            # Return 10 nodes for flexibility
            result.stdout = "\n".join([f"node{i:02d}" for i in range(1, 11)])
            result.returncode = 0
            return result
        raise subprocess.CalledProcessError(1, cmd)

    with patch("subprocess.run", side_effect=mock_run):
        yield


class TestRecipeLoading:
    """Test that all recipe files load correctly."""

    @pytest.mark.parametrize("recipe_path", get_recipe_files(), ids=lambda p: p.name)
    def test_recipe_loads(self, recipe_path, mock_slurm_env, mock_scontrol):
        """Each recipe file loads without error."""
        from srtctl.core.config import load_config

        config = load_config(str(recipe_path))

        # Basic sanity checks
        assert config.name is not None
        assert config.model.path is not None
        assert config.resources.gpu_type is not None


class TestRuntimeContextCreation:
    """Test RuntimeContext creation with mocked SLURM."""

    def test_nodes_from_slurm(self, mock_slurm_env, mock_scontrol):
        """Nodes are correctly parsed from SLURM environment."""
        from srtctl.core.runtime import Nodes

        nodes = Nodes.from_slurm()
        assert nodes.head == "node01"
        assert nodes.bench == "node01"
        assert len(nodes.worker) == 10

    @pytest.mark.skipif(not get_recipe_files(), reason="No recipe files found")
    def test_runtime_context_from_first_recipe(self, mock_slurm_env, mock_scontrol):
        """RuntimeContext is created correctly from a real recipe."""
        from srtctl.core.config import load_config
        from srtctl.core.runtime import RuntimeContext

        recipe = get_recipe_files()[0]
        config = load_config(str(recipe))

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = RuntimeContext.from_config(
                config, job_id="12345", log_dir_base=Path(tmpdir)
            )

            assert runtime.job_id == "12345"
            assert runtime.run_name == f"{config.name}_12345"
            assert runtime.nodes.head == "node01"
            assert runtime.log_dir.exists()


class TestEndpointAllocation:
    """Test endpoint allocation for various configs."""

    @pytest.mark.skipif(not get_recipe_files(), reason="No recipe files found")
    def test_disaggregated_allocation(self, mock_slurm_env, mock_scontrol):
        """Disaggregated configs correctly allocate endpoints."""
        from srtctl.core.config import load_config
        from srtctl.core.runtime import RuntimeContext

        # Find a disaggregated recipe (has prefill_nodes and decode_nodes)
        disagg_recipes = []
        for recipe in get_recipe_files():
            config = load_config(str(recipe))
            if config.resources.prefill_nodes and config.resources.decode_nodes:
                disagg_recipes.append((recipe, config))

        if not disagg_recipes:
            pytest.skip("No disaggregated recipes found")

        recipe, config = disagg_recipes[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = RuntimeContext.from_config(
                config, job_id="12345", log_dir_base=Path(tmpdir)
            )

            r = config.resources
            endpoints = config.backend.allocate_endpoints(
                num_prefill=r.num_prefill,
                num_decode=r.num_decode,
                num_agg=r.num_agg,
                gpus_per_prefill=r.gpus_per_prefill,
                gpus_per_decode=r.gpus_per_decode,
                gpus_per_agg=r.gpus_per_agg,
                gpus_per_node=r.gpus_per_node,
                available_nodes=runtime.nodes.worker,
            )

            # Check we have correct number of endpoints
            expected = r.num_prefill + r.num_decode
            assert len(endpoints) == expected

            # Check prefill/decode split
            prefill_eps = [e for e in endpoints if e.mode == "prefill"]
            decode_eps = [e for e in endpoints if e.mode == "decode"]
            assert len(prefill_eps) == r.num_prefill
            assert len(decode_eps) == r.num_decode

    @pytest.mark.skipif(not get_recipe_files(), reason="No recipe files found")
    def test_aggregated_allocation(self, mock_slurm_env, mock_scontrol):
        """Aggregated configs correctly allocate endpoints."""
        from srtctl.core.config import load_config
        from srtctl.core.runtime import RuntimeContext

        # Find an aggregated recipe (has agg_workers, no prefill/decode)
        agg_recipes = []
        for recipe in get_recipe_files():
            config = load_config(str(recipe))
            if config.resources.agg_workers and not config.resources.prefill_nodes:
                agg_recipes.append((recipe, config))

        if not agg_recipes:
            pytest.skip("No aggregated recipes found")

        recipe, config = agg_recipes[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = RuntimeContext.from_config(
                config, job_id="12345", log_dir_base=Path(tmpdir)
            )

            r = config.resources
            endpoints = config.backend.allocate_endpoints(
                num_prefill=r.num_prefill,
                num_decode=r.num_decode,
                num_agg=r.num_agg,
                gpus_per_prefill=r.gpus_per_prefill,
                gpus_per_decode=r.gpus_per_decode,
                gpus_per_agg=r.gpus_per_agg,
                gpus_per_node=r.gpus_per_node,
                available_nodes=runtime.nodes.worker,
            )

            # All endpoints should be agg mode
            assert all(e.mode == "agg" for e in endpoints)
            assert len(endpoints) == r.agg_workers


class TestCommandGeneration:
    """Test command generation for workers and benchmarks."""

    @pytest.mark.skipif(not get_recipe_files(), reason="No recipe files found")
    def test_worker_command_generation(self, mock_slurm_env, mock_scontrol):
        """Worker commands are generated correctly."""
        from srtctl.core.config import load_config
        from srtctl.core.runtime import RuntimeContext

        recipe = get_recipe_files()[0]
        config = load_config(str(recipe))

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = RuntimeContext.from_config(
                config, job_id="12345", log_dir_base=Path(tmpdir)
            )

            r = config.resources
            endpoints = config.backend.allocate_endpoints(
                num_prefill=r.num_prefill,
                num_decode=r.num_decode,
                num_agg=r.num_agg,
                gpus_per_prefill=r.gpus_per_prefill,
                gpus_per_decode=r.gpus_per_decode,
                gpus_per_agg=r.gpus_per_agg,
                gpus_per_node=r.gpus_per_node,
                available_nodes=runtime.nodes.worker,
            )
            processes = config.backend.endpoints_to_processes(endpoints)

            # Build command for first process
            proc = processes[0]
            cmd = config.backend.build_worker_command(
                process=proc,
                endpoint_processes=processes,
                runtime=runtime,
                use_sglang_router=config.frontend.use_sglang_router,
            )

            # Verify key command components
            cmd_str = " ".join(cmd)
            assert "python3" in cmd_str or "dynamo" in cmd_str
            assert "--model-path" in cmd_str or "model-path" in cmd_str

    @pytest.mark.skipif(not get_recipe_files(), reason="No recipe files found")
    def test_benchmark_command_generation(self, mock_slurm_env, mock_scontrol):
        """Benchmark commands are generated correctly."""
        from srtctl.benchmarks import get_runner
        from srtctl.core.config import load_config
        from srtctl.core.runtime import RuntimeContext

        # Find a recipe with sa-bench
        for recipe in get_recipe_files():
            config = load_config(str(recipe))
            if config.benchmark.type == "sa-bench":
                break
        else:
            pytest.skip("No sa-bench recipes found")

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = RuntimeContext.from_config(
                config, job_id="12345", log_dir_base=Path(tmpdir)
            )

            runner = get_runner("sa-bench")

            # Validate config
            errors = runner.validate_config(config)
            assert errors == [], f"Validation errors: {errors}"

            # Build command
            cmd = runner.build_command(config, runtime)

            assert "bash" in cmd[0]
            assert "sa-bench" in cmd[1]
            assert "http://localhost:8000" in cmd[2]


class TestFullOrchestrationFlow:
    """Test the full orchestration flow with mocked srun."""

    @pytest.mark.skipif(not get_recipe_files(), reason="No recipe files found")
    def test_orchestrator_initialization(self, mock_slurm_env, mock_scontrol):
        """Orchestrator initializes correctly."""
        from srtctl.cli.do_sweep import SweepOrchestrator
        from srtctl.core.config import load_config
        from srtctl.core.runtime import RuntimeContext

        recipe = get_recipe_files()[0]
        config = load_config(str(recipe))

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = RuntimeContext.from_config(
                config, job_id="12345", log_dir_base=Path(tmpdir)
            )

            orchestrator = SweepOrchestrator(config, runtime)

            # Verify orchestrator has correct attributes
            assert orchestrator.config == config
            assert orchestrator.runtime == runtime
            assert orchestrator.backend == config.backend

    @pytest.mark.skipif(not get_recipe_files(), reason="No recipe files found")
    def test_endpoint_and_process_computation(self, mock_slurm_env, mock_scontrol):
        """Orchestrator correctly computes endpoints and processes."""
        from srtctl.cli.do_sweep import SweepOrchestrator
        from srtctl.core.config import load_config
        from srtctl.core.runtime import RuntimeContext

        recipe = get_recipe_files()[0]
        config = load_config(str(recipe))

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = RuntimeContext.from_config(
                config, job_id="12345", log_dir_base=Path(tmpdir)
            )

            orchestrator = SweepOrchestrator(config, runtime)

            # Access cached properties
            endpoints = orchestrator.endpoints
            processes = orchestrator.backend_processes

            # Should have computed both
            assert len(endpoints) > 0
            assert len(processes) > 0
            # Processes >= endpoints (multi-node TP means 1 endpoint -> N processes)
            assert len(processes) >= len(endpoints)


class TestContainerMounts:
    """Test container mount generation."""

    @pytest.mark.skipif(not get_recipe_files(), reason="No recipe files found")
    def test_default_mounts(self, mock_slurm_env, mock_scontrol):
        """Default mounts are created correctly."""
        from srtctl.core.config import load_config
        from srtctl.core.runtime import RuntimeContext

        recipe = get_recipe_files()[0]
        config = load_config(str(recipe))

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = RuntimeContext.from_config(
                config, job_id="12345", log_dir_base=Path(tmpdir)
            )

            mounts = runtime.container_mounts

            # Model should be mounted at /model
            assert Path("/model") in mounts.values()

            # Logs should be mounted at /logs
            assert Path("/logs") in mounts.values()

    @pytest.mark.skipif(not get_recipe_files(), reason="No recipe files found")
    def test_mount_string_generation(self, mock_slurm_env, mock_scontrol):
        """Container mount string is generated correctly."""
        from srtctl.core.config import load_config
        from srtctl.core.runtime import RuntimeContext

        recipe = get_recipe_files()[0]
        config = load_config(str(recipe))

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = RuntimeContext.from_config(
                config, job_id="12345", log_dir_base=Path(tmpdir)
            )

            mount_str = runtime.get_container_mounts_str()

            # Should be comma-separated host:container pairs
            assert ":" in mount_str
            assert "/model" in mount_str
            assert "/logs" in mount_str


class TestBenchmarkRunners:
    """Test all benchmark runners with real recipes."""

    @pytest.mark.skipif(not get_recipe_files(), reason="No recipe files found")
    def test_all_recipe_benchmarks_validate(self, mock_slurm_env, mock_scontrol):
        """All recipes have valid benchmark configs for their type."""
        from srtctl.benchmarks import get_runner
        from srtctl.core.config import load_config
        from srtctl.core.runtime import RuntimeContext

        for recipe in get_recipe_files():
            config = load_config(str(recipe))

            # Skip manual benchmarks
            if config.benchmark.type == "manual":
                continue

            try:
                runner = get_runner(config.benchmark.type)
            except ValueError:
                # Unknown benchmark type, skip
                continue

            errors = runner.validate_config(config)
            assert errors == [], f"{recipe.name}: {errors}"

    @pytest.mark.skipif(not get_recipe_files(), reason="No recipe files found")
    def test_all_recipe_benchmarks_build_command(self, mock_slurm_env, mock_scontrol):
        """All recipes can build benchmark commands."""
        from srtctl.benchmarks import get_runner
        from srtctl.core.config import load_config
        from srtctl.core.runtime import RuntimeContext

        for recipe in get_recipe_files():
            config = load_config(str(recipe))

            # Skip manual benchmarks
            if config.benchmark.type == "manual":
                continue

            try:
                runner = get_runner(config.benchmark.type)
            except ValueError:
                continue

            with tempfile.TemporaryDirectory() as tmpdir:
                runtime = RuntimeContext.from_config(
                    config, job_id="12345", log_dir_base=Path(tmpdir)
                )

                # Should not raise
                cmd = runner.build_command(config, runtime)
                assert len(cmd) > 0
                assert "bash" in cmd[0]
