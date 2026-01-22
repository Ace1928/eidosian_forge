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
def plotLists(cnt):
    result = []
    clr = ['#DD1E2F', '#EBB035', '#06A2CB', '#218559', '#D0C6B1', '#192823', '#DDAACC']
    for i in range(1, cnt):
        list, errList = getLists(data, i)
        result.append(ax.bar(ticks + width * i, list, width, color=clr[i - 1], yerr=errList))
    return result