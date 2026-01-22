import base64
import json
import numpy as np
import pandas as pd
from panel.pane import (
from panel.tests.util import streamz_available
def test_json_pane_rerenders_on_depth_change(document, comm):
    pane = JSON({'a': 2}, depth=2)
    model = pane.get_root(document, comm=comm)
    pane.depth = -1
    assert model.depth is None