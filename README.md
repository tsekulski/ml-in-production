# ml-in-production
Demo of preparing a machine learning model for production deployment with the use of:
* sklearn pipeline (incl. custom transformers)
* Flask

## Getting Started

<These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.>

### Prerequisites

* Anaconda Python 3.x and Jupyter Notebooks
* https://www.anaconda.com/distribution/

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

8. Install packages from `src` directory:
```
pip install -e .
```

9. Add conda environment to jupyter kernels:
```
python -m ipykernel install --user --name ml-in-production --display-name "ml-in-production"
```

10.

(End with an example of getting some data out of the system or using it for a little demo)

## Running the tests

(Explain how to run the automated tests for this system)

### Break down into end to end tests

(Explain what these tests test and why)

```
Give an example
```

### And coding style tests

(Explain what these tests test and why)

```
Give an example
```

## Deployment

(Add additional notes about how to deploy this on a live system)

## Built With

* [Anaconda](https://www.anaconda.com) - Python distribution and Jupyter Notebooks