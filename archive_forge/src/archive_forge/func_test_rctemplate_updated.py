import os
import numpy as np
import pytest
from xarray.core.indexing import MemoryCachedArray
from ...data import datasets, load_arviz_data
from ...rcparams import (
from ...stats import compare
from ..helpers import models  # pylint: disable=unused-import
def test_rctemplate_updated():
    fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../arvizrc.template')
    rc_pars_template = read_rcfile(fname)
    rc_defaults = rc_params(ignore_files=True)
    assert all((key in rc_pars_template.keys() for key in rc_defaults.keys())), [key for key in rc_defaults.keys() if key not in rc_pars_template]
    assert all((value == rc_pars_template[key] for key, value in rc_defaults.items())), [key for key, value in rc_defaults.items() if value != rc_pars_template[key]]