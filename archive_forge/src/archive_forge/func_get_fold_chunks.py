from parlai.core.teachers import (
from parlai.core.opt import Opt
import copy
import random
import itertools
import os
from PIL import Image
import string
import json
from abc import ABC
from typing import Tuple, List
def get_fold_chunks(self, opt) -> List[int]:
    datatype = opt['datatype']
    if 'train' in datatype:
        return list(range(50))
    elif 'valid' in datatype:
        return list(range(50, 60))
    elif 'test' in datatype:
        return list(range(60, 70))