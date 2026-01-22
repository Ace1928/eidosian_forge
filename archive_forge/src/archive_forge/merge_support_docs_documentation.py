import os
import json
from parlai.core.params import ParlaiParser
from os.path import join as pjoin
from os.path import isdir, isfile
from glob import glob
from data_utils import merge_support_docs

File adapted from
https://github.com/facebookresearch/ELI5/blob/master/data_creation/merge_support_docs.py
Modified to use data directory rather than a hard-coded processed data directory.
