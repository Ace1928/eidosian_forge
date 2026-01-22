import os
from os import path as op
import string
import errno
from glob import glob
import nibabel as nb
import imghdr
from .base import (
def sanitize_path_comp(path_comp):
    result = []
    for char in path_comp:
        if char not in string.ascii_letters + string.digits + '-_.':
            result.append('_')
        else:
            result.append(char)
    return ''.join(result)