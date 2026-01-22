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
def step_0_get_chembl_data(self):
    """Download Compound-Data for self.acc_id, these are available in self.sd_entries afterwards"""

    def looks_like_number(x):
        """Check for proper Float-Value"""
        try:
            float(x)
            return True
        except ValueError:
            return False
    if self.acc_id.find('CHEMBL') == -1:
        self.target_data = requests.get('https://www.ebi.ac.uk/chemblws/targets/uniprot/{}.json'.format(self.acc_id), proxies=self.proxy).json()
    else:
        self.target_data = {}
        self.target_data['target'] = {}
        self.target_data['target']['chemblId'] = self.acc_id
    self.chembl_id = self.target_data['target']['chemblId']
    self.request_data['chembl_id'] = self.target_data['target']['chemblId']
    self.bioactivity_data = requests.get('https://www.ebi.ac.uk/chemblws/targets/{}/bioactivities.json'.format(self.target_data['target']['chemblId']), proxies=self.proxy).json()
    ic50_skip = 0
    ki_skip = 0
    inhb_skip = 0
    count = 0
    non_homo = 0
    self.dr = {}
    i = 0
    x = len(self.bioactivity_data['bioactivities'])
    for bioactivity in [record for record in self.bioactivity_data['bioactivities'] if looks_like_number(record['value'])]:
        if i % 100 == 0:
            sys.stdout.write('\r' + str(i) + '/' + str(x) + ' >          <\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08')
        elif i % 100 % 10 == 0:
            sys.stdout.write('|')
        sys.stdout.flush()
        i += 1
        if bioactivity['organism'] != 'Homo sapiens':
            non_homo += 1
            continue
        if re.search('IC50', bioactivity['bioactivity_type']):
            if bioactivity['units'] != 'nM':
                ic50_skip += 1
                continue
        elif re.search('Ki', bioactivity['bioactivity_type']):
            ki_skip += 1
            continue
        elif re.search('Inhibition', bioactivity['bioactivity_type']):
            inhb_skip += 1
        else:
            continue
        self.cmpd_data = requests.get('https://www.ebi.ac.uk/chemblws/compounds/{}.json'.format(bioactivity['ingredient_cmpd_chemblid']), proxies=self.proxy).json()
        my_smiles = self.cmpd_data['compound']['smiles']
        bioactivity['Smiles'] = my_smiles
        self.dr[count] = bioactivity
        count += 1
    SDtags = self.dr[0].keys()
    cpd_counter = 0
    self.sd_entries = []
    for x in range(len(self.dr)):
        entry = self.dr[x]
        cpd = Chem.MolFromSmiles(str(entry['Smiles']))
        AllChem.Compute2DCoords(cpd)
        cpd.SetProp('_Name', str(cpd_counter))
        cpd_counter += 1
        for tag in SDtags:
            cpd.SetProp(str(tag), str(entry[tag]))
        self.sd_entries.append(cpd)
    return True