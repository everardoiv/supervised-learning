Everardo Villasenor (evillasenor3)
Assignment 1
Code found at: https://github.com/everardoiv/supervised-learning
----------------------------------------

This code was built on Python 3.6
The following packages are required to run the experiments:
- Numpy
- Pandas
- csv
- matplotlib
- sklearn

----------------------------------------

In order to run this code you will need to be using a Python 3.6 environment with the list of dependecies having been installed (pip install X).

The two datsets are included as csv files. They are: 'data/output/credit-card-data.csv' and 'data/output/sampled-poker-hand-data.csv'

The file structure used was the following:
- supervised_learning.py
--/analysis/
---*-analysis.txt
--/data/
---/output/
----credit-card-data.csv
----sampled-poker-hand-data.csv
-/figures/
--*.png

Within supervised_learning.py, a function called experiment_inputs defines what experiment to run. It is recommended to only run on one dataset at a time and comment or uncomment out the appropriate models. The parameters to search over are found in the function set_parameters. Note that this is different from the dict of params_to_test which defines what parameters will be visualized with a validation curve. The parameters in set_parameters is the grid of hyper-parameters to search over, which can increase experimentation time.

The classification reports and metrics measures are enclosed under the analysis folder and are generated after running the models. 

In order to run the code you must change directories to the parent folder and run (example):
python supervised_learning.py
