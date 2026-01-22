import os
import numpy as np
import pytest
from ... import from_emcee
from ..helpers import _emcee_lnprior as emcee_lnprior
from ..helpers import _emcee_lnprob as emcee_lnprob
from ..helpers import (  # pylint: disable=unused-import
def test_verify_var_names(self, data):
    with pytest.raises(ValueError):
        from_emcee(data.obj, var_names=['not', 'enough'])