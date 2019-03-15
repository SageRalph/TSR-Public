# TSR Evaluation Toolkit

This repository includes all the data and code necessary to replicate the results in the paper "Recommendations from Cold Starts in Big Data".
This includes the full Isle of Wight Supply Chain (IWSC) dataset, a full implementation of the Transitive Semantic Relationships (TSR) inference algorithm, and scripts for automated evaluation using both explicit and implicit feedback. Please refer to the original paper for detailed information about the IWSC dataset and TSR approach.

## Citations
For all derivative works using the IWSC dataset or TSR algorithm, please cite the original paper "Recommendations from Cold Starts in Big Data".

## Requirements ##

All code is targeted for Python 3.6

A fast multi-core CPU and at least 2GB free memory is strongly recommended for implicit evaluation.

Pre-computed description embeddings for IWSC are provided to minimise dependencies and requirements.
TensorFlow or similar may be needed to test alternative embeddings.


## Setup ##

The recommended installation method is using Pip and Pipenv.  
A full list of dependencies can be found in the `Pipfile` file if manual installation is preferred. 

1. Install [Python 3.6](https://www.python.org/) and [Pip](https://pip.pypa.io/en/latest/).

2. Install pipenv

        $ pip install pipenv

3. Navigate to the repo's directory

4. Install dependencies using pipenv

        $ pipenv install


## How to run ##

Run any of the Python scripts using pipenv

        $ pipenv run python (script name)

All commands used to produce the results in the `results` directory can be found in `commands.txt`.


# Dataset files

`datasets/IWSC.json` is a full copy of the Isle of Wight Supply Chain (IWSC) dataset.  
If publishing results using this dataset please reference the original paper "Recommendations from Cold Starts in Big Data".

`datasets/IWSC.USEDAN.json` is a copy of IWSC with added description embeddings generated using [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/2).


# Results files

This repository contains all the data and code nessesary to replicate our results.  
We provide the full results presented in the paper "Recommendations from Cold Starts in Big Data" in the `results` directory.  
All results provided use the dataset `datasets/IWSC.USEDAN.json`.

`results/tasks.explicit.csv` contains the results of evaluation of TSR-a on all IWSC tasks using `scripts/TSREvalExplicit.py`.

`results/tasks.implicit.csv` contains the results of evaluation of TSR-a on all IWSC tasks using `scripts/TSREvalImplicit.py`.

`results/algorithms.explicit.csv` contains the results of evaluation of all TSR scoring algorithms on the IWSC SL_consumers task using `scripts/TSREvalExplicit.py`.

`results/algorithms.explicit.csv` contains the results of evaluation of all TSR scoring algorithms on the IWSC SL_consumers task using `scripts/TSREvalImplicit.py`.


# Scripts

All scripts are self-documented and viewers are encouraged to verify our approach.  
A summary of each script is provided here for convenience.

Note: Scripts set to output results to an existing file will append the results. This way multiple commands can target the same file for comparison of results.

### TSREvalExplicit.py
This script uses TSRCore to perform leave-one-out cross validation of 
the TSR inference algorithm. Most options can be specified by commandline.
Scoring uses R-Precision and error values.

### TSREvalImplicit.py
This script uses TSRCore to perform implicit feedback 1-in-100 evaluation of
the TSR inference algorithm. Most options can be specified by commandline.
1-in-100 evaluation groups one known positive with 100 random items and scores
based on the algorithm's ability to rank the true positive highly.  
Note that as batches created are randomly, results may vary over multiple runs. 
A greater repeat count will give more consistent results.  
This script can be time and resource intensive if the repeat count is high.  
This script is optimised for multi-core CPUs.


### TSRCore.py
This script contains an implementation of the TSR inference algorithm.
If publishing results using any variation of this approach please reference the original paper "Recommendations from Cold Starts in Big Data".

### util.py
This script contains common functions for data handling and evaluation.
