import csv 
import json
import sys
import os
import requests
from pprint import pprint
import random
import time
import xlsxwriter
from random import gauss
import math
import sqlite3
from copy import deepcopy
from icecream import ic
import operator
import pickle

import statistics 
from statistics import mean

import pandas as pd
from pandas import DataFrame

import scipy.stats as stats
from scipy.stats import ranksums

from itertools import combinations
from itertools import permutations
from itertools import product

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import numpy as np
from numpy import array
from numpy import transpose
from numpy import std
from numpy import argmax
from numpy import any as numpy_any
from numpy import all as numpy_all
from numpy import sum
from numpy import sqrt
from numpy import int64
from numpy import mean as np_mean
from numpy import max as np_max
from numpy import savetxt
from numpy import squeeze
from numpy import asarray
from numpy import argsort as argsort
from numpy import matrix as np_matrix
from numpy import arange as arange
from numpy import seterr

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

# All scorer objects follow the convention that higher return values are better than lower return values. Thus metrics which measure the distance between the model and the data, like metrics.mean_squared_error, are available as neg_mean_squared_error which return the negated value of the metric.
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
    precision_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    recall_score,
    log_loss,
    roc_auc_score,
    precision_recall_fscore_support,
    zero_one_loss,
    
    brier_score_loss,
    jaccard_score,
    cohen_kappa_score,
    hinge_loss,
    matthews_corrcoef,
    hamming_loss,
    
    confusion_matrix,
    multilabel_confusion_matrix,
    precision_recall_curve,
    roc_curve
)

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.utils import check_array
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

import keras
from keras import layers














