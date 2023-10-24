import sys
import os
import sqlite3
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
import numpy as np
from numpy import savetxt
from copy import deepcopy
from icecream import ic
from numpy import transpose
from statistics import mean
import csv
from numpy import array
from numpy import delete
import math
from numpy import argsort as argsort
from past.builtins import execfile
import subprocess
import tensorflow as tf
import random


# Get parameters given for script execution.
# The 1st parameter is the dataset name to use. It is the name of the file without the extension. The dataset must be a .txt file with ";" as delimiter.
dataset = ""
try:
    dataset = sys.argv[1]
except:
    print("ERROR: You must give a dataset name to use as first parameter. Just give the name without the .txt extension.")
    sys.exit()

# Get Raw Data.
# df must have as last column a categorical variable to predict ("y" variable) (in bmdec it is a combination of transport modes) and as other columns various predictor categorical variables ("x" variables).
df = []
with open("Data/" + dataset + ".txt", newline='') as txtfile:
        df = list(csv.reader(txtfile, delimiter=';'))

# Transform df into a Dataframe.
df = DataFrame(df)
df = df.values

# Scale of the criteria = Scale of df's columns except the first.
# MinMaxScaler() scales values to have them laying between 0 and 1 included.
# scaler = preprocessing.MinMaxScaler()
# df[:,1:] = scaler.fit_transform(df[:,1:])

# Save df as a file.
pd.DataFrame(df).to_csv("Outputs/df.csv", header=None, index=None)

# Parameters to call main.py
# Path to df.csv
path_to_df = "Outputs/df.csv"
# Macro iterations.
macro_iterations = 1
# Algorithms to execute. Also used for files saved.
# Type inside brackets the names of the algorithms you want to execute. 
# Available algorithms : "naive_bayes", "decision_tree", "logistic_regression", "neural_net_categorical", "k_nearest_neighbours", "support_vector_machine", "csp"
# Not suitable for categorical predictions : "linear_regression", "neural_network", "k_means"
algos = ["csp"]
algo_types = ""
for t in algos:
    algo_types += t + "-"
algo_types = algo_types[:-1] # remove last character ("-")

# Call main.py with its parameters.
#os.system("3_main.py " + path_to_df + " " + str(macro_iterations) + " " + str(algo_types))
subprocess.run(["python", "3_main.py", path_to_df, str(macro_iterations), str(algo_types)])
