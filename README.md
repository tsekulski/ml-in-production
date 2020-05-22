# ml-in-production
# Preparing machine learning models for real-time scoring in production environment with sklearn pipelines and Flask

## The notebook available in the `/notebooks` directory gives an overview of the following:
### --> Typical sequence of data cleaning, feature engineering, model training steps in a machine learning project
### --> The challenge of trying to reproduce these steps in real-time in the live scoring environment
### --> Standard sklearn pipeline: how can it help us address this challenge - but only partly
### --> Customizing sklearn pipeline to solve our challenge in an e2e fashion
### --> Using Flask to serve a ML model as a REST API

## Getting Started

### Prerequisites

* Anaconda Python 3.x and Jupyter Notebooks
* https://www.anaconda.com/distribution/

The project has been tested with the following system and dependencies:
```
Operating system: Linux
OS release: 4.15.0-101-generic
Machine: x86_64
Platform: Linux-4.15.0-101-generic-x86_64-with-debian-buster-sid
Version: #102-Ubuntu SMP Mon May 11 10:07:26 UTC 2020

Python version: 3.7.7 (default, May  7 2020, 21:25:33) 
[GCC 7.3.0]
Pandas version: 0.24.2
Numpy version: 1.16.3
Scikit-learn version: 0.20.3
```

### Installing

1. Create a conda virtual environment:
```
conda create --name ml-in-production
```

2. Activate the environment:
```
conda activate ml-in-production
```
or if your shell doesn't support the `conda` command:
```
source activate ml-in-production
```

7. Install dependencies as specified in the `requirements.txt` file:
```
conda install --file requirements.txt
```

8. Install packages from `src` directory (inside the activated conda environment):
```
pip install -e .
```

9. Add conda environment to jupyter kernels:
```
python -m ipykernel install --user --name ml-in-production --display-name "ml-in-production"
```

10. Change directory to project's directory and start jupyter server:
```
jupyter notebook
```

11. Run the notebook available in the `/notebooks` directory