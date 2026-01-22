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
def load_from_chunk(self, chunk_idx: int):
    output = []
    for i in range(10):
        text = ' '.join([str(i)] + [str(chunk_idx)] * 5)
        resp = ' '.join([str(i)])
        output.append((text, resp))
    return output