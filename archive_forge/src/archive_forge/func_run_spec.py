from statsmodels.compat.pandas import deprecate_kwarg
import os
import subprocess
import tempfile
import re
from warnings import warn
import pandas as pd
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (X13NotFoundError,
def run_spec(x12path, specpath, outname=None, meta=False, datameta=False):
    if meta and datameta:
        raise ValueError('Cannot specify both meta and datameta.')
    if meta:
        args = [x12path, '-m ' + specpath]
    elif datameta:
        args = [x12path, '-d ' + specpath]
    else:
        args = [x12path, specpath]
    if outname:
        args += [outname]
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)