import torch
import numpy as np
import scipy.sparse as sp
import argparse
import os
import math
from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import Counter
from . import utils
from .doc_db import DocDB
from . import tokenizers
import parlai.utils.logging as logging
def get_doc_freqs_t(cnts):
    """
    Return word --> # of docs it appears in (torch version).
    """
    return torch.histc(cnts._indices()[0].float(), bins=cnts.size(0), min=0, max=cnts.size(0))