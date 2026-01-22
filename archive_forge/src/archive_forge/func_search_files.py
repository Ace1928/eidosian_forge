import os
import re
from copy import deepcopy
import itertools as it
import glob
from glob import iglob
from ..utils.filemanip import split_filename
from .base import (
def search_files(prefix, outtypes):
    return it.chain.from_iterable((iglob(glob.escape(prefix + outtype)) for outtype in outtypes))