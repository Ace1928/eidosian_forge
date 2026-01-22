import numpy as np
import pandas as pd
import pytest
from unittest import TestCase, SkipTest
from hvplot.util import (
def test_fn_kwds(self):
    import panel as pn
    from ..util import process_dynamic_args
    x = 'sepal_length'
    y = 'sepal_width'
    kind = 'scatter'
    by_species = pn.widgets.Checkbox(name='By species')
    color = pn.widgets.ColorPicker(value='#ff0000')

    @pn.depends(by_species.param.value, color.param.value)
    def by_species_fn(by_species, color):
        return 'species' if by_species else color
    dynamic, arg_deps, arg_names = process_dynamic_args(x, y, kind, c=by_species_fn)
    assert dynamic == {}
    assert arg_names == ['c', 'c']
    assert len(arg_deps) == 2