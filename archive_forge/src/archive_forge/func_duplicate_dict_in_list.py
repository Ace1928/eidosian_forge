import json
import os
import numpy as np
def duplicate_dict_in_list(dict, list):
    for item in list:
        if recursive_dict_eq(dict, item):
            return True
    return False