import datetime
import json
import numpy as np
from ase.utils import reader, writer
def fix_int_keys_in_dicts(obj):
    """Convert "int" keys: "1" -> 1.

    The json.dump() function will convert int keys in dicts to str keys.
    This function goes the other way.
    """
    if isinstance(obj, dict):
        return {intkey(key): fix_int_keys_in_dicts(value) for key, value in obj.items()}
    return obj