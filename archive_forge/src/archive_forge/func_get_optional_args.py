import os
import re
import sys
import numpy as np
import inspect
import sysconfig
def get_optional_args(self, func):
    signature = inspect.signature(func)
    optional_args = {}
    for k, v in signature.parameters.items():
        if v.default is not inspect.Parameter.empty:
            optional_args[k] = v.default
    return optional_args