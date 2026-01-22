from collections import OrderedDict, deque
import numpy as np
from ray.rllib.utils import force_list
from ray.rllib.utils.framework import try_import_tf
def nb_common_elem(l1, l2):
    return len([e for e in l1 if e in l2])