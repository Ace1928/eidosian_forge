import pytest
import numpy as np
from traitlets import TraitError, Undefined
from ..ndarray.traits import shape_constraints
from ..ndarray.widgets import (
def test_datawidget_creation_blank():
    with pytest.raises(TraitError):
        w = NDArrayWidget()