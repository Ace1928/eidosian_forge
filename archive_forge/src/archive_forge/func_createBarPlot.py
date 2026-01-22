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
def createBarPlot(data):

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

    def plotLists(cnt):
        result = []
        clr = ['#DD1E2F', '#EBB035', '#06A2CB', '#218559', '#D0C6B1', '#192823', '#DDAACC']
        for i in range(1, cnt):
            list, errList = getLists(data, i)
            result.append(ax.bar(ticks + width * i, list, width, color=clr[i - 1], yerr=errList))
        return result
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 6)
    ticks = np.arange(0.0, 12.0, 1.2)
    if len(self.model) == 1:
        ticks = np.arange(0.0, 1.0, 1.5)
    width = 0.15
    plots = plotLists(8)
    ax.set_xticks(ticks + 0.75)
    ax.set_xticklabels([str(x) for x in range(1, 11, 1)])
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('# model')
    ax.set_xlim(-0.3, 14)
    ax.set_ylim(-0.1, 1.2)
    ax.legend(tuple(plots), [x for x in data[0][1:8]], 'upper right')
    best, worst = findBestWorst(data)
    if len(self.model) > 1:
        ax.annotate('best', xy=(ticks[best], 0.85), xytext=(ticks[best] + 0.25, 1.1), color='green')
        ax.annotate('worst', xy=(ticks[worst], 0.85), xytext=(ticks[worst] + 0.25, 1.1), color='red')
    fig.savefig('barplot.png', transparent=True)
    return