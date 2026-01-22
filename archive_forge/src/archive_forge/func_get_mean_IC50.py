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
def get_mean_IC50(mol_list):
    IC50 = 0
    IC50_avg = 0
    for bla in mol_list:
        try:
            IC50 += float(bla.GetProp('value'))
        except Exception:
            print('no IC50 reported', bla.GetProp('_Name'))
    IC50_avg = IC50 / len(mol_list)
    return IC50_avg