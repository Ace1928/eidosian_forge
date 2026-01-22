from glob import glob
import os
from ... import logging
from ...utils.filemanip import fname_presuffix
from ..base import traits, isdefined, CommandLine, CommandLineInputSpec, PackageInfo
from ...external.due import BibTeX
def no_fsl():
    """Checks if FSL is NOT installed
    used with skipif to skip tests that will
    fail if FSL is not installed"""
    if Info.version() is None:
        return True
    else:
        return False