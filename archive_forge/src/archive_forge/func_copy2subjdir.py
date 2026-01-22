import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
def copy2subjdir(cls, in_file, folder=None, basename=None, subject_id=None):
    """Method to copy an input to the subjects directory"""
    if not isdefined(in_file):
        return in_file
    if isdefined(cls.inputs.subjects_dir):
        subjects_dir = cls.inputs.subjects_dir
    else:
        subjects_dir = os.getcwd()
    if not subject_id:
        if isdefined(cls.inputs.subject_id):
            subject_id = cls.inputs.subject_id
        else:
            subject_id = 'subject_id'
    if basename is None:
        basename = os.path.basename(in_file)
    if folder is not None:
        out_dir = os.path.join(subjects_dir, subject_id, folder)
    else:
        out_dir = os.path.join(subjects_dir, subject_id)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, basename)
    if not os.path.isfile(out_file):
        shutil.copy(in_file, out_file)
    return out_file