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
def step_3_merge_IC50(self):
    """merge IC50 of duplicates into one compound using mean of all values if:
        min(IC50) => IC50_avg-3*IC50_stddev && max(IC50) <= IC50_avg+3*IC50_stddev && IC50_stddev <= IC50_avg"""
    np_old_settings = np.seterr(invalid='ignore')

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

    def get_stddev_IC50(mol_list):
        IC50_list = []
        for mol in mol_list:
            try:
                IC50_list.append(round(float(mol.GetProp('value')), 2))
            except Exception:
                print('no IC50 reported', mol.GetProp('_Name'))
        IC50_stddev = np.std(IC50_list, ddof=1)
        return (IC50_stddev, IC50_list)
    result = []
    IC50_dict = {}
    for cpd in self.sd_entries:
        if 'cansmirdkit' not in cpd.GetPropNames():
            Chem.RemoveHs(cpd)
            cansmi = Chem.MolToSmiles(cpd, canonical=True)
            cpd.SetProp('cansmirdkit', cansmi)
        cansmi = str(cpd.GetProp('cansmirdkit'))
        IC50_dict[cansmi] = {}
    for cpd in self.sd_entries:
        cansmi = str(cpd.GetProp('cansmirdkit'))
        try:
            IC50_dict[cansmi].append(cpd)
        except Exception:
            IC50_dict[cansmi] = [cpd]
    for entry in IC50_dict:
        IC50_avg = str(get_mean_IC50(IC50_dict[entry]))
        IC50_stddev, IC50_list = get_stddev_IC50(IC50_dict[entry])
        IC50_dict[entry][0].SetProp('value_stddev', str(IC50_stddev))
        IC50_dict[entry][0].SetProp('value', IC50_avg)
        minimumvalue = float(IC50_avg) - 3 * float(IC50_stddev)
        maximumvalue = float(IC50_avg) + 3 * float(IC50_stddev)
        if round(IC50_stddev, 1) == 0.0:
            result.append(IC50_dict[entry][0])
        elif IC50_stddev > float(IC50_avg):
            runawaylist = []
            for e in IC50_dict[entry]:
                runawaylist.append(e.GetProp('_Name'))
                print('stddev larger than mean', runawaylist, IC50_list, IC50_avg, IC50_stddev)
        elif np.min(IC50_list) < minimumvalue or np.max(IC50_list) > maximumvalue:
            pass
        else:
            result.append(IC50_dict[entry][0])
    self.sd_entries = result
    np.seterr(over=np_old_settings['over'], divide=np_old_settings['divide'], invalid=np_old_settings['invalid'], under=np_old_settings['under'])
    return True