import os
import re
import shutil
import sys
import tempfile
import zipfile
from glob import glob
from os.path import abspath
from os.path import join as pjoin
from subprocess import PIPE, Popen
import os
import sys
import {mod_name}
def zip_extract_all(fname, path=None):
    """Extract all members from zipfile

    Deals with situation where the directory is stored in the zipfile as a name,
    as well as files that have to go into this directory.
    """
    zf = zipfile.ZipFile(fname)
    members = zf.namelist()
    members = [m for m in members if not m.endswith('/')]
    for zipinfo in members:
        zf.extract(zipinfo, path, None)