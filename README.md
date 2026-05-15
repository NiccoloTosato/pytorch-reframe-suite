# ReFrame PyTorch benchmark suite

This repository contains a small set of ReFrame tests that benchmark:

- **Training throughput** with `torch.distributed` (DDP)
- **Collective communication** (currently all-reduce bandwidth)

The primary entry point for these is [run_pytorch.py](run_pytorch.py).

## What the tests do

### 1) Training benchmark (DDP)

The training tests inherit from `PyTorchTestBase` in [pytorch_test_base.py](pytorch_test_base.py).


- Reports two performance metrics:
	- `samples_per_sec_per_gpu`: mean of per-epoch “images/sec”
	- `samples_per_sec_total`: mean of the final “Total average”

 `PyTorchDdpPipAmd` in [run_pytorch.py](run_pytorch.py) adds:

- A per-run Python virtualenv and pip-installed ROCm PyTorch
- An aws-ofi network plugin build (see below)


### 2) Collective benchmark (all-reduce)

`PyTorchRCCLAllReduceVenv` in [run_pytorch.py](run_pytorch.py) downloads and runs the IBM
PyTorch communication microbenchmark (`allreduce-stats.py`) via `torchrun`.

- Fetches the benchmark script at runtime using `curl`
- Uses `torchrun` rendezvous over Slurm nodes to run multi-node all-reduce
- Extracts bandwidth results from the benchmark output and reports:
	- `avg_bw` (GB/s)
	- `max_bw` (GB/s)

## ROCm versions - Important note

This suite intentionally distinguishes between two ROCm versions, because they are used for
different purposes and often have different formatting:

- `rocm_version_pytorch` (example: `7.2`)
	- Used to select the **PyTorch wheel index**:
		`https://download.pytorch.org/whl/rocm<rocm_version_pytorch>`
- `rocm_version_module` (example: `7.2.0`)
	- Used to load the **system ROCm modules** (e.g., `module load rocm/<ver>`)
	- Used as the key for selecting which aws-ofi implementation to build

Both are defined as ReFrame variables in `VersionsHandlerPlugin` (see [run_pytorch.py](run_pytorch.py)),
and can be overridden from the ReFrame CLI via `-S`.

## aws-ofi selection (based on `rocm_version_module`)

The aws-ofi plugin is fetched and built as part of the environment setup:

- If `rocm_version_module >= 7.1.0`:
	- Clone `https://github.com/aws/aws-ofi-nccl.git`
	- Checkout the requested `aws-ofi-nccl` release tag (default: `1.19.1`)
- Else (fallback / legacy):
	- Clone `https://github.com/ROCm/aws-ofi-rccl.git` (deprecated)
	- Checkout the `cxi` branch

The selection logic lives in `FetchAws.prepare_download()` in [run_pytorch.py](run_pytorch.py).

## Runtime PyTorch version discovery

At import time, [run_pytorch.py](run_pytorch.py) dynamically queries the PyTorch ROCm wheel index
to discover the most recent `torch==X.Y.Z` versions available for the selected `rocm_version_pytorch`.

Because this happens before ReFrame instantiates the tests, the module uses a small CLI-scraping helper
(`GetCliVar()` in [utils.py](utils.py)) to read `-S rocm_version_pytorch=...` / `-S rocm_version_module=...`
directly from `sys.argv` during the import phase.

- Implementation: `GetLatestPytorch()` in [utils.py](utils.py)
- The discovered versions are used to parameterize the venv setup test via `make_test(...)`
	(so the suite can test multiple recent PyTorch versions without hard-coding them).



## How to run

Example :

`reframe -C sysconfig.yaml -c run_pytorch.py --system=odo:batch \
	-S rocm_version_pytorch="7.2" -S rocm_version_module="7.2.0" \
	--job-option='--time="1:0:0"' -r`

Run only the training test:

`reframe -C sysconfig.yaml -c run_pytorch.py --system=odo:batch -n PyTorchDdpPipAmd -r`

Run only the all-reduce test:

`reframe -C sysconfig.yaml -c run_pytorch.py --system=odo:batch -n PyTorchRCCLAllReduceVenv -r`
