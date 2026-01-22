import re
from warnings import warn
from nltk.corpus import bcp47
def inverse_dict(dic):
    """Return inverse mapping, but only if it is bijective"""
    if len(dic.keys()) == len(set(dic.values())):
        return {val: key for key, val in dic.items()}
    else:
        warn('This dictionary has no bijective inverse mapping.')