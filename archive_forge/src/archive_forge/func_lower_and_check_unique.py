from __future__ import annotations
import re
from collections import defaultdict
import numpy as np
def lower_and_check_unique(dict_to_check):
    """
    Takes a dictionary and makes all the keys lower case. Also converts all numeric
    values (floats, ints) to str and replaces "jobtype" with "job_type" just so that
    key specifically can be called elsewhere without ambiguity. Finally, ensures that
    multiple identical keys, that differed only due to different capitalizations, are not
    present. If there are multiple equivalent keys, an Exception is raised.

    Args:
        dict_to_check (dict): The dictionary to check and standardize

    Returns:
        to_return (dict): An identical dictionary but with all keys made
            lower case and no identical keys present.
    """
    if dict_to_check is None:
        return None
    to_return = {}
    for key, val in dict_to_check.items():
        new_key = key.lower()
        if isinstance(val, str):
            val = val.lower()
        elif isinstance(val, (int, float)):
            val = str(val)
        else:
            pass
        if new_key == 'jobtype':
            new_key = 'job_type'
        if new_key in to_return and val != to_return[new_key]:
            raise ValueError(f'Multiple instances of key {new_key} found with different values! Exiting...')
        to_return[new_key] = val
    return to_return