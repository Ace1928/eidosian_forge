from glob import glob
import os
from ... import logging
from ...utils.filemanip import fname_presuffix
from ..base import traits, isdefined, CommandLine, CommandLineInputSpec, PackageInfo
from ...external.due import BibTeX
def no_fsl_course_data():
    """check if fsl_course data is present"""
    return not ('FSL_COURSE_DATA' in os.environ and os.path.isdir(os.path.abspath(os.environ['FSL_COURSE_DATA'])))