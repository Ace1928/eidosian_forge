import argparse
import gc
import logging
import os
import sys
import traceback
import types
import time
import json
from pyomo.common.deprecation import deprecated
from pyomo.common.log import is_debug_set
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.fileutils import import_file
from pyomo.common.tee import capture_output
from pyomo.common.dependencies import (
from pyomo.common.collections import Bunch
from pyomo.opt import ProblemFormat
from pyomo.opt.base import SolverFactory
from pyomo.opt.parallel import SolverManagerFactory
from pyomo.dataportal import DataPortal
from pyomo.scripting.interface import (
from pyomo.core import Model, TransformationFactory, Suffix, display
def pyomo_excepthook(etype, value, tb):
    """
        This exception hook gets called when debugging is on. Otherwise,
        run_command in this module is called.
        """
    global filter_excepthook
    if len(data.options.model.filename) > 0:
        name = 'model ' + data.options.model.filename
    else:
        name = 'model'
    if filter_excepthook:
        action = 'loading'
    else:
        action = 'running'
    msg = 'Unexpected exception (%s) while %s %s:\n    ' % (etype.__name__, action, name)
    valueStr = str(value)
    if etype == KeyError:
        valueStr = valueStr.replace('\\n', '\n')
        if valueStr[0] == valueStr[-1] and valueStr[0] in '"\'':
            valueStr = valueStr[1:-1]
    logger.error(msg + valueStr, extra={'cleandoc': False})
    tb_list = traceback.extract_tb(tb, None)
    i = 0
    if not is_debug_set(logger) and filter_excepthook:
        while i < len(tb_list):
            if data.options.model.filename in tb_list[i][0]:
                break
            i += 1
        if i == len(tb_list):
            i = 0
    print('\nTraceback (most recent call last):')
    for item in tb_list[i:]:
        print('  File "' + item[0] + '", line ' + str(item[1]) + ', in ' + item[2])
        if item[3] is not None:
            print('    ' + item[3])
    sys.exit(1)