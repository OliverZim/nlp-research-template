# An opinionated template for NLP research code

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

NLP research template for training language models from scratch using PyTorch + PyTorch Lightning + Weights & Biases + HuggingFace. It's built to be customized but provides comprehensive, sensible default functionality.

## Setup

### Preliminaries

It's recommended to use [`mamba`](https://github.com/mamba-org/mamba) to manage dependencies. `mamba` is a drop-in replacement for `conda` re-written in C++ to speed things up significantly (you can stick with `conda` though). To provide reproducible environments, we use `conda-lock` to generate lockfiles for each platform.

<details><summary>Installing <code>mamba</code></summary>

<p>

On Unix-like platforms, run the snippet below. Otherwise, visit the [mambaforge repo](https://github.com/conda-forge/miniforge#mambaforge). Note this does not use the Anaconda installer, which reduces bloat.

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```

</details>

<details><summary>Installing <code>conda-lock</code></summary>

<p>

The preferred method is to install conda-lock into your `mamba` / `conda` `base` environment using `mamba install -c conda-forge -n base conda-lock`. Then, you can access conda-lock via the automatic subcommand discovery (e.g. `mamba lock --version`). Otherwise, visit the [conda-lock repo](https://github.com/conda/conda-lock).

```bash
mamba lock install --name gpt4 --file conda-lock.yml # create environment based on lockfile
mamba lock # create new lockfile based on environment.yml
mamba lock --update # update packages in lockfile
```

</details>

### Environment

After having installed `mamba` and `conda-lock`, you can create a `mamba` environment from a lockfile with all necessary dependencies installed like this:

```bash
mamba lock install --name <gpt4> --file conda-lock.yml
```

That's it -- this is the power of lockfiles.

To generate new lockfiles after updating the `environment.yml` file, simply run `mamba lock`.

<details><summary>Setup on <code>ppc64le</code></summary>

<p>
It's slightly more tricky because the official channels do not provide packages compiled for `ppc64le`. However, we can use the amazing [Open-CE channel](https://opence.mit.edu/#/) by MIT instead.

```bash
mamba create -n gpt4 python=3.10 && mamba activate gpt4
mamba install pytorch cudatoolkit -c https://opence.mit.edu -c conda-forge -c defaults
```

</p>
</details>

## Training

To start a language model MLM training, run:

```bash
python train.py --data /path/to/data/dir --model roberta-base --gpus 2 --offline
```

By default, `train.txt` and `dev.txt` are expected in the data directory. To see an overview over all options and their defaults, run `python train.py --help`.
We have disabled Weights & Biases syncing with the `--offline` flag. To enable W&B, enter your `WANDB_ENTITY` and `WANDB_PROJECT` in [dlib/frameworks/wandb.py](dlib/frameworks/wandb.py) and simply omit the `--offline` flag.
