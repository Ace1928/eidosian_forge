from collections import abc as container_abcs, defaultdict
from copy import deepcopy
from itertools import chain
import torch
import bitsandbytes.functional as F
def update_group(group, new_group):
    new_group['params'] = group['params']
    return new_group