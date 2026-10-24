# Developing Tunix on one GPU or CPU

Tunix is optimized for TPUs, but you do not need one to work on it. This guide
shows how to set up a development environment on a laptop or a single-GPU
workstation, run the CPU tests that run on every pull request, and build the
documentation.

It does not cover TPU or multi-host setups, or the vLLM and SGLang-JAX samplers,
which are TPU-only installs (see the [Quickstart](../quickstart.md)).

## Supported platforms

The CPU tests in CI run on Ubuntu with Python 3.12
(`.github/workflows/cpu-tests.yml`). Use Linux, or Windows through WSL2.

```{note}
Native Windows is not supported at the moment. `jax[tpu]` is a hard dependency
of `google-tunix`, and its `libtpu` package is only published for Linux, so
`pip install -e .` cannot resolve on Windows (see
[issue #2047](https://github.com/google/tunix/issues/2047)). Under WSL2, clone
the repository into the Linux file system (for example `~/src/tunix`), not
under `/mnt/c`.
```

## Prerequisites

- Python 3.12 or newer.

- A C++ compiler, CMake and Ninja. Installing Tunix builds a few small
  [nanobind](https://nanobind.readthedocs.io) extensions. On Ubuntu or Debian:

  ```sh
  sudo apt-get update
  sudo apt-get install -y build-essential cmake ninja-build git
  ```

- For a GPU, an NVIDIA driver that is recent enough for the CUDA version of the
  JAX wheel you install. See the
  [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for
  the details.

## Create the environment

From a clone of the repository:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[test,experimental,docs]"
```

The editable install compiles the extensions and places them next to the Python
sources. The `test` extra provides pytest, the `experimental` extra is needed by
the tests under `tests/experimental/`, and the `docs` extra is needed to build
this documentation.

## Choose a JAX backend

`pip install` brings `jax[tpu]` (and `libtpu`) with it. You select the backend
that JAX actually uses with the `JAX_PLATFORMS` environment variable.

**CPU**

```sh
export JAX_PLATFORMS=cpu
python -c "import jax; print(jax.devices())"
```

This prints a single CPU device.

**One NVIDIA GPU**

Install the CUDA build of JAX, staying below the upper bound that
`pyproject.toml` sets for `jax`:

```sh
pip install -U "jax[cuda13]<0.11.1"
export JAX_PLATFORMS=cuda
python -c "import jax; print(jax.devices())"
```

This prints one CUDA device. Use `jax[cuda12]` if your driver does not support
CUDA 13. By default JAX preallocates 75% of the GPU memory. If you share the GPU
with a desktop session or other processes, set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` (see
[GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)).

## Run the tests

Start with a small test that builds a tiny model and samples from it:

```sh
export JAX_PLATFORMS=cpu
python -m pytest tests/generate/sampler_test.py -k test_samples -v
```

Pull requests are checked by the `run_core` job in
`.github/workflows/cpu-tests.yml`. That file is the source of truth for the
exact commands. The main ones are:

```sh
python -m pytest tests/rl/agentic/ -v --tb=short

python -m pytest tests/rl/ -v --tb=short \
  -k "not test_grpo_with_lora_model" \
  --ignore=tests/rl/agentic --ignore=tests/rl/rl_cluster_test.py

python -m pytest tests/cli/utils/ -v --tb=short

python -m pytest tests/utils/mesh_test.py tests/utils/topology_test.py \
  -v --tb=short

python -m pytest tests/perf/ -v --tb=short

python -m pytest tests/models/ -v --tb=short \
  --ignore=tests/models/naming_test.py
```

Run the subset that covers the code you changed. A few things are not part of
this CPU workflow:

- The tests that need a TPU, or vLLM and `tpu-inference`.
- The `run_dev` job, which downloads models and needs Hugging Face and Kaggle
  tokens. It is skipped for pull requests from forks.
- The `tests/dist_rl/` and `tests/experimental/` suites need the gRPC protobuf
  definitions compiled first. See the "Compile gRPC Protobuf definitions" step
  in `cpu-tests.yml`.

## Build the documentation

Documentation warnings are treated as errors in CI. Run the build from the
`docs/` directory:

```sh
cd docs
sphinx-build -W -b html . _build/html
```

Open `docs/_build/html/index.html` in a browser to check the result.

## Format and lint

Install the pre-commit hooks once, and run them on the files you changed:

```sh
pip install pre-commit
pre-commit install
pre-commit run --files path/to/changed_file.py
```

See [Contributing](../contributing.md) for the formatting and review process.
