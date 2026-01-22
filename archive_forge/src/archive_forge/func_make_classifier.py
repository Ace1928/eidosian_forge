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
def make_classifier(featuresets):
    return WekaClassifier.train('/tmp/name.model', featuresets, 'C4.5')