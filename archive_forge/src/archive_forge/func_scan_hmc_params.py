import json
import math
import re
from typing import Any, Dict, List, MutableMapping, Optional, TextIO, Union
import numpy as np
import pandas as pd
from cmdstanpy import _CMDSTAN_SAMPLING, _CMDSTAN_THIN, _CMDSTAN_WARMUP
def scan_hmc_params(fd: TextIO, config_dict: Dict[str, Any], lineno: int) -> int:
    """
    Scan step size, metric from  stan_csv file comment lines.
    """
    metric = config_dict['metric']
    line = fd.readline().strip()
    lineno += 1
    if not line == '# Adaptation terminated':
        raise ValueError('line {}: expecting metric, found:\n\t "{}"'.format(lineno, line))
    line = fd.readline().strip()
    lineno += 1
    label, step_size = line.split('=')
    if not label.startswith('# Step size'):
        raise ValueError('line {}: expecting step size, found:\n\t "{}"'.format(lineno, line))
    try:
        float(step_size.strip())
    except ValueError as e:
        raise ValueError('line {}: invalid step size: {}'.format(lineno, step_size)) from e
    before_metric = fd.tell()
    line = fd.readline().strip()
    lineno += 1
    if metric == 'unit_e':
        if line.startswith('# No free parameters'):
            return lineno
        else:
            fd.seek(before_metric)
            return lineno - 1
    if not (metric == 'diag_e' and line == '# Diagonal elements of inverse mass matrix:' or (metric == 'dense_e' and line == '# Elements of inverse mass matrix:')):
        raise ValueError('line {}: invalid or missing mass matrix specification'.format(lineno))
    line = fd.readline().lstrip(' #\t')
    lineno += 1
    num_unconstrained_params = len(line.split(','))
    if metric == 'diag_e':
        return lineno
    else:
        for _ in range(1, num_unconstrained_params):
            line = fd.readline().lstrip(' #\t')
            lineno += 1
            if len(line.split(',')) != num_unconstrained_params:
                raise ValueError('line {}: invalid or missing mass matrix specification'.format(lineno))
        return lineno