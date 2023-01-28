# Fairify

This repository contains the source code, benchmark models, and datasets for the paper - **"Fairify: Fairness Verification of Neural Networks"**, appeared in ICSE 2023 at Melbourne, Australia.

### Authors
* Sumon Biswas, Carnegie Mellon University (sumonb@cs.cmu.edu)
* Hridesh Rajan, Iowa State University (hridesh@iastate.edu)

**PDF** https://arxiv.org/abs/2212.06140

**DOI:** This artifact is also published in Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7579939.svg)](https://doi.org/10.5281/zenodo.7579939)

![The problem tackled by Fairify](/problem.jpg)

## Index
> 1. [Models](models/)
> 2. Datasets
  >> * [German Credit (GC)](data/german)
  >> * [Adult Census (AC)](data/adult)
  >> * [Bank Marketing (BM)](data/bank)
> 3. Verification source code
  >> * [Verify models](src/)
  >> * [Verify models with scaled experiment](stress/)
  >> * [Verify relaxed queries](relaxed/)
  >> * Verify relaxed queries
  >  >> * [Targeted queries 1](targeted/)
  >  >> * [Targeted queries 2](targeted2/)
  >> * [Utilities](utils/)
> 4. Appendix
  >> * [Supplementary results](/Appendix-Result.pdf)

## Installation

To run Fairify, we need to install Python 3 environment. The current version has been tested on Python 3.7. It is recommended to install Python virtual environment for the tool. Furthermore, we used bash shell scripts to automate running benchmark and Python scripts. Below are step-by-step instructions to setup environment and run the tool. 

### Environment Setup

Follow these steps to create a virtual environment and clone the Fairify repository.

2. Clone this repository and move to the directory:

```
git clone https://github.com/sumonbis/Fairify
cd Fairify/
``` 

1. Run this on command line to create a virtual environment:

```
python3 -m venv fenv
source fenv/bin/activate
```

Run the following command to update pip on Python: `python3 -m pip install --upgrade pip`. Alternatively, you can follow the [Python documentation](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) to install virtual environment on your machine. 

3. Navigate to the cloned repository: `cd Fairify/` and install required packages:

```
pip install -r requirements.txt
```

To run the tool, please refer to the [installation file](/INSTALL.md) for detailed instructions. 

### Cite the paper as
```
@inproceedings{biswas23fairify,
  author = {Sumon Biswas and Hridesh Rajan},
  title = {Fairify: Fairness Verification of Neural Networks},
  booktitle = {ICSE'23: The 45th International Conference on Software Engineering},
  location = {Melbourne, Australia},
  month = {May 14-May 20},
  year = {2023},
  entrysubtype = {conference}
}
```