import csv
import gzip
import json
import math
import optparse
import os
import pickle
import re
import sys
from pickle import Unpickler
import numpy as np
import requests
from pylab import *
from scipy import interp, stats
from sklearn import cross_validation, metrics, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (auc, make_scorer, precision_score, recall_score,
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, SDWriter
from rdkit.ML.Descriptors import MoleculeDescriptors
from the one dimensional weights.
def getLists(data, col):
    accList = []
    errList = []
    for x in data[1:]:
        if x[col].find('_') == -1:
            continue
        if x[col].find('.pkl') != -1:
            continue
        spl = x[col].split('_')
        accList.append(float(spl[0]))
        errList.append(float(spl[1]))
    return (accList, errList)