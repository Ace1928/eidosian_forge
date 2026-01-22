import os
from sys import platform
import shutil
from ... import logging, LooseVersion
from ...utils.filemanip import split_filename, fname_presuffix
from ..base import (
from ...external.due import BibTeX
@staticmethod
def standard_image(img_name):
    """
        Grab an image from the standard location.

        Could be made more fancy to allow for more relocatability

        """
    clout = CommandLine('which afni', ignore_exception=True, resource_monitor=False, terminal_output='allatonce').run()
    if clout.runtime.returncode != 0:
        return None
    out = clout.runtime.stdout
    basedir = os.path.split(out)[0]
    return os.path.join(basedir, img_name)