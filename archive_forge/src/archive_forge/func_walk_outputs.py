import os
import sys
import pickle
from collections import defaultdict
import re
from copy import deepcopy
from glob import glob
from pathlib import Path
from traceback import format_exception
from hashlib import sha1
from functools import reduce
import numpy as np
from ... import logging, config
from ...utils.filemanip import (
from ...utils.misc import str2bool
from ...utils.functions import create_function_from_source
from ...interfaces.base.traits_extension import (
from ...interfaces.base.support import Bunch, InterfaceResult
from ...interfaces.base import CommandLine
from ...interfaces.utility import IdentityInterface
from ...utils.provenance import ProvStore, pm, nipype_ns, get_id
from inspect import signature
def walk_outputs(object):
    """Extract every file and directory from a python structure"""
    out = []
    if isinstance(object, dict):
        for _, val in sorted(object.items()):
            if isdefined(val):
                out.extend(walk_outputs(val))
    elif isinstance(object, (list, tuple)):
        for val in object:
            if isdefined(val):
                out.extend(walk_outputs(val))
    elif isdefined(object) and isinstance(object, (str, bytes)):
        if os.path.islink(object) or os.path.isfile(object):
            out = [(filename, 'f') for filename in get_all_files(object)]
        elif os.path.isdir(object):
            out = [(object, 'd')]
    return out