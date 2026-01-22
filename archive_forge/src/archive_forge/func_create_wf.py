import os
from copy import deepcopy
import pytest
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces import utility as niu
from .... import config
from ..utils import (
def create_wf(name):
    """Creates a workflow for the following tests"""

    def fwhm(fwhm):
        return fwhm
    pipe = pe.Workflow(name=name)
    process = pe.Node(niu.Function(input_names=['fwhm'], output_names=['fwhm'], function=fwhm), name='proc')
    process.iterables = ('fwhm', [0])
    process2 = pe.Node(niu.Function(input_names=['fwhm'], output_names=['fwhm'], function=fwhm), name='proc2')
    process2.iterables = ('fwhm', [0])
    pipe.connect(process, 'fwhm', process2, 'fwhm')
    return pipe