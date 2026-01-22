import json
import os
import numpy as np
def recursive_dict_eq(d1, d2):
    if isinstance(d1, dict) != isinstance(d2, dict):
        return False
    if isinstance(d1, dict):
        if set(d1.keys()) != set(d2.keys()):
            return False
        return all([recursive_dict_eq(d1[k], d2[k]) for k in d1.keys()])
    else:
        return d1 == d2