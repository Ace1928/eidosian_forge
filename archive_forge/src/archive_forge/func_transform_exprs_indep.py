from __future__ import (absolute_import, division, print_function)
from functools import reduce
import inspect
import math
import operator
import sys
from pkg_resources import parse_requirements, parse_version
import numpy as np
import pytest
def transform_exprs_indep(fw, bw, dep_exprs, indep, check=True):
    """ Transform x in dydx

    Parameters
    ----------
    fw: expression
        forward transformation
    bw: expression
        backward transformation
    dep_exprs: iterable of (symbol, expression) pairs
        pairs of (dependent variable, derivative expressions)
    check: bool (default: True)
        whether to verification of the analytic correctness should
        be performed

    Returns
    -------
    List of transformed expressions for dydx
    """
    if check:
        if fw.subs(indep, bw) - indep != 0:
            fmtstr = 'Incorrect (did you set real=True?) fw: %s'
            raise ValueError(fmtstr % str(fw))
        if bw.subs(indep, fw) - indep != 0:
            fmtstr = 'Incorrect (did you set real=True?) bw: %s'
            raise ValueError(fmtstr % str(bw))
    dep, exprs = zip(*dep_exprs)
    return [(e / fw.diff(indep)).subs(indep, bw) for e in exprs]