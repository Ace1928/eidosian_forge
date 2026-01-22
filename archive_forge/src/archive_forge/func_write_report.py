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
def write_report(node, report_type=None, is_mapnode=False):
    """Write a report file for a node - DEPRECATED"""
    if report_type not in ('preexec', 'postexec'):
        logger.warning('[Node] Unknown report type "%s".', report_type)
        return
    write_node_report(node, is_mapnode=is_mapnode, result=node.result if report_type == 'postexec' else None)