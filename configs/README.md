# Configs Directory

This directory contains configuration files and binaries used during benchmark runs.

## Contents

- `deepep_config.json` - Dynamo configuration (committed to repo)
- Downloaded by `make setup`:
  - `nats-server` - NATS binary for message queue
  - `etcd` / `etcdctl` - etcd binaries for distributed coordination
  - `ai_dynamo*.whl` - Python wheels for Dynamo runtime

## Usage

During SLURM job execution, this directory is mounted as `/config` inside containers, making all files accessible to the benchmark workloads.

## Setup

Run `make setup` to download required binaries and dependencies.
