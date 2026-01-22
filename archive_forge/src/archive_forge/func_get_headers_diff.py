import hashlib
import os
import re
import sys
from collections import OrderedDict
from optparse import Option, OptionParser
import numpy as np
import nibabel as nib
import nibabel.cmdline.utils
def get_headers_diff(file_headers, names=None):
    """Get difference between headers

    Parameters
    ----------
    file_headers: list of actual headers (dicts) from files
    names: list of header fields to test

    Returns
    -------
    dict
      str: list for each header field which differs, return list of
      values per each file
    """
    difference = OrderedDict()
    fields = names
    if names is None:
        fields = file_headers[0].keys()
    for field in fields:
        values = [header.get(field) for header in file_headers]
        if are_values_different(*values):
            difference[field] = values
    return difference