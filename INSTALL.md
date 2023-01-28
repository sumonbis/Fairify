# Installation and Usage

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

### Run the Fairify tool

#### Experiment 1
Navigate to the src directory `cd src/` with command line. To run verification on all the models, run the following command. Here, `dataset` is any of the three: `AC`, `GC`, or `BM`. All the models in one dataset are run with the following command:

```
./fairify.sh <dataset>
```

**Example:** To run the verification for all the 8 models in German Credit, run `./fairify.sh GC`. Each model is configured to be run for 30 minutes. The above command automatically runs for all the models in the corresponding dataset. The raw results will be generated in the `res` directory inside each dataset directory. All the results from our experiments are included in `csv` files.

#### Experiment 2
Not all the models produce SAT or UNSAT in the given time. The rest of the models are again verified using the scaled experiment setup, which is located in `stress` folder. The instruction to run the verification is same as above: navigate to stress directory: `cd stress/` and run the following command. Here, each model takes 1 hour.

```
./fairify-stress.sh <dataset>
```

**Example:** To run the German Credit model (1 model), run `./fairify-stress.sh GC` when you are in `stress` directory. This will take 1 hour to finish. The other dataset models could be run in the same way. 

#### Experiment 3

The relaxed query verification are also in the corresponding separate directory - `relaxed`. The verification for those experiments can be run in the same way. 

```
./fairify-relaxed.sh <dataset>
```

**Example:** Navigate using `cd relaxed/` and then run `./fairify-relaxed.sh GC`. Since there are 5 models in this dataset, it will take 5 hours. However, the results are generated in the corresponding `csv` file after each dataset finishes verifying (1 hour). 

#### Experiment 4

Finally, the verification source code and results for the first and second targeted verification queries are in the `targeted` and `targeted2` directory. Navigate to one of these two directories and run the following command:

```
./fairify-targeted.sh <dataset>
```

**Example:** Navigate using `cd targeted/`, and then run `./fairify-targeted.sh GC`. This will take 1 hour for each model, in total 5 hours. 