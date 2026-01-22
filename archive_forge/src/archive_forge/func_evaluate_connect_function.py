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
def evaluate_connect_function(function_source, args, first_arg):
    func = create_function_from_source(function_source)
    try:
        output_value = func(first_arg, *list(args))
    except NameError as e:
        if e.args[0].startswith('global name') and e.args[0].endswith('is not defined'):
            e.args = (e.args[0], 'Due to engine constraints all imports have to be done inside each function definition')
        raise e
    return output_value