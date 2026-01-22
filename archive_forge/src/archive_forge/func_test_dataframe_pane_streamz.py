import base64
import json
import numpy as np
import pandas as pd
from panel.pane import (
from panel.tests.util import streamz_available
@streamz_available
def test_dataframe_pane_streamz(document, comm):
    from streamz.dataframe import Random
    sdf = Random(interval='200ms', freq='50ms')
    pane = DataFrame(sdf)
    assert pane._stream is None
    model = pane.get_root(document, comm=comm)
    assert pane._stream is not None
    assert pane._models[model.ref['id']][0] is model
    assert model.text == ''
    pane.object = sdf.x
    assert pane._models[model.ref['id']][0] is model
    assert model.text == ''
    pane._cleanup(model)
    assert pane._stream is None
    assert pane._models == {}