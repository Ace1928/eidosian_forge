import os
from os import path as op
import string
import errno
from glob import glob
import nibabel as nb
import imghdr
from .base import (
def make_key_func(meta_keys, index=None):

    def key_func(src_nii):
        result = [src_nii.get_meta(key, index) for key in meta_keys]
        return result
    return key_func