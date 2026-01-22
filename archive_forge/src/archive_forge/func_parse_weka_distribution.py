import os
import re
import subprocess
import tempfile
import time
import zipfile
from sys import stdin
from nltk.classify.api import ClassifierI
from nltk.internals import config_java, java
from nltk.probability import DictionaryProbDist
def parse_weka_distribution(self, s):
    probs = [float(v) for v in re.split('[*,]+', s) if v.strip()]
    probs = dict(zip(self._formatter.labels(), probs))
    return DictionaryProbDist(probs)