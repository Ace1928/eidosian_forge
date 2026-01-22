from __future__ import annotations
from pathlib import Path
import argparse
import enum
import sys
import stat
import time
import abc
import platform, subprocess, operator, os, shlex, shutil, re
import collections
from functools import lru_cache, wraps, total_ordering
from itertools import tee
from tempfile import TemporaryDirectory, NamedTemporaryFile
import typing as T
import textwrap
import pickle
import errno
import json
from mesonbuild import mlog
from .core import MesonException, HoldableObject
from glob import glob
def get_filenames_templates_dict(inputs: T.List[str], outputs: T.List[str]) -> T.Dict[str, T.Union[str, T.List[str]]]:
    """
    Create a dictionary with template strings as keys and values as values for
    the following templates:

    @INPUT@  - the full path to one or more input files, from @inputs
    @OUTPUT@ - the full path to one or more output files, from @outputs
    @OUTDIR@ - the full path to the directory containing the output files

    If there is only one input file, the following keys are also created:

    @PLAINNAME@ - the filename of the input file
    @BASENAME@ - the filename of the input file with the extension removed

    If there is more than one input file, the following keys are also created:

    @INPUT0@, @INPUT1@, ... one for each input file

    If there is more than one output file, the following keys are also created:

    @OUTPUT0@, @OUTPUT1@, ... one for each output file
    """
    values: T.Dict[str, T.Union[str, T.List[str]]] = {}
    if inputs:
        values['@INPUT@'] = inputs
        for ii, vv in enumerate(inputs):
            values[f'@INPUT{ii}@'] = vv
        if len(inputs) == 1:
            values['@PLAINNAME@'] = plain = os.path.basename(inputs[0])
            values['@BASENAME@'] = os.path.splitext(plain)[0]
    if outputs:
        values['@OUTPUT@'] = outputs
        for ii, vv in enumerate(outputs):
            values[f'@OUTPUT{ii}@'] = vv
        values['@OUTDIR@'] = os.path.dirname(outputs[0])
        if values['@OUTDIR@'] == '':
            values['@OUTDIR@'] = '.'
    return values