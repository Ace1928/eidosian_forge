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
def write_node_report(node, result=None, is_mapnode=False):
    """Write a report file for a node."""
    if not str2bool(node.config['execution']['create_report']):
        return
    cwd = node.output_dir()
    report_file = Path(cwd) / '_report' / 'report.rst'
    report_file.parent.mkdir(exist_ok=True, parents=True)
    lines = [write_rst_header('Node: %s' % get_print_name(node), level=0), write_rst_list(['Hierarchy : %s' % node.fullname, 'Exec ID : %s' % node._id]), write_rst_header('Original Inputs', level=1), write_rst_dict(node.inputs.trait_get())]
    if result is None:
        logger.debug('[Node] Writing pre-exec report to "%s"', report_file)
        report_file.write_text('\n'.join(lines), encoding='utf-8')
        return
    logger.debug('[Node] Writing post-exec report to "%s"', report_file)
    lines += [write_rst_header('Execution Inputs', level=1), write_rst_dict(node.inputs.trait_get()), write_rst_header('Execution Outputs', level=1)]
    outputs = result.outputs
    if outputs is None:
        lines += ['None']
        report_file.write_text('\n'.join(lines), encoding='utf-8')
        return
    if isinstance(outputs, Bunch):
        lines.append(write_rst_dict(outputs.dictcopy()))
    elif outputs:
        lines.append(write_rst_dict(outputs.trait_get()))
    else:
        lines += ['Outputs object was empty.']
    if is_mapnode:
        lines.append(write_rst_header('Subnode reports', level=1))
        nitems = len(ensure_list(getattr(node.inputs, node.iterfield[0])))
        subnode_report_files = []
        for i in range(nitems):
            subnode_file = Path(cwd) / 'mapflow' / ('_%s%d' % (node.name, i)) / '_report' / 'report.rst'
            subnode_report_files.append('subnode %d : %s' % (i, subnode_file))
        lines.append(write_rst_list(subnode_report_files))
        report_file.write_text('\n'.join(lines), encoding='utf-8')
        return
    lines.append(write_rst_header('Runtime info', level=1))
    rst_dict = {'hostname': result.runtime.hostname, 'duration': result.runtime.duration, 'working_dir': result.runtime.cwd, 'prev_wd': getattr(result.runtime, 'prevcwd', '<not-set>')}
    for prop in ('cmdline', 'mem_peak_gb', 'cpu_percent'):
        if hasattr(result.runtime, prop):
            rst_dict[prop] = getattr(result.runtime, prop)
    lines.append(write_rst_dict(rst_dict))
    if hasattr(result.runtime, 'merged'):
        lines += [write_rst_header('Terminal output', level=2), write_rst_list(result.runtime.merged)]
    if hasattr(result.runtime, 'stdout'):
        lines += [write_rst_header('Terminal - standard output', level=2), write_rst_list(result.runtime.stdout)]
    if hasattr(result.runtime, 'stderr'):
        lines += [write_rst_header('Terminal - standard error', level=2), write_rst_list(result.runtime.stderr)]
    if hasattr(result.runtime, 'environ'):
        lines += [write_rst_header('Environment', level=2), write_rst_dict(result.runtime.environ)]
    report_file.write_text('\n'.join(lines), encoding='utf-8')