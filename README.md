# Fairify

This repository contains the source code, benchmark models, and datasets for the paper - "Fairify: Fairness Verification of Neural Networks", appeared in ICSE 2023 at Melbourne, Australia.

### Authors
* Sumon Biswas, Carnegie Mellon University
* Hridesh Rajan, Iowa State University

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

![The problem tackled by Fairify](/problem.jpg)

To run Fairify, we need to install Python 3 environment. The current version has been tested on Python 3.7. It is recommended to install Python virtual environment for the tool.

First, clone this repository. Fairify uses Z3 as the SMT solver. The other required Python packages are listed in [requirements.txt](/requirements.txt). To run verification on all the models, navigate to `src` directory in terminal and run the following command. Here, `dataset` is any of the three: AC, GC, or BM.

```
python Verify-<dataset>.py
```

For example, to run the verification for all the 8 models in German Credit, run `python Verify-GC.py`. Each model is configured to be run for 30 minutes. The above automatically runs for all the models in the corresponding dataset. The raw results will be generated in the `res` directory inside each dataset directory. All the results from our experiments are include in `csv` files.

The models which could not be verified in the above steps are again verified using the scaled experiment setup, located in `stress` folder. The instruction to run the verification is same as above.

The verification for the relaxed queries are also in the corresponding separate directory - `relaxed`. The verification for those experiments can be run in the same way. Note that, depending on the configuration of the machine, the number of partitions that are verified should vary.

Finally, the verification source code and results for the first and second targeted verification queries are in the `targeted` and `targeted2` directory.

We leveraged Numpy arrays and matrix operations to enable tracking the NN structure (e.g., layers, neurons) and perform network pruning. Below is a Python function of canonical NN with one input-hidden-output layer. 

```python
def net(x, w, b): # Inputs are Numpy arrays for input data, weight and bias
  # Input layer
  x1 = w[0].T @ x + b[0] # WS calculation using matrix multiplication
  y1 = numpy.maximum(0, x1) # ReLU activation
  # Hidden layer
  x2 = w[1].T @ y1 + b[1]
  y2 = numpy.maximum(0, x2) # ReLU activation
  # ... Output layer(s)
  x_out = w[2].T @ y2 + b[2]
  y = 1 / (1 + math.exp(-x_out)) # Sigmoid activation
  return y # Output of NN
```

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