import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
def createoutputdirs(outputs):
    """create all output directories. If not created, some freesurfer interfaces fail"""
    for output in list(outputs.values()):
        dirname = os.path.dirname(output)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)