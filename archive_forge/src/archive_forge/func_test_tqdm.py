import time
import numpy as np
import pandas as pd
import pytest
from tqdm.contrib.concurrent import process_map
import panel as pn
from panel.widgets import Tqdm
def test_tqdm():
    tqdm = Tqdm(layout='row', sizing_mode='stretch_width')
    for _ in tqdm(range(3)):
        pass
    assert tqdm.value == 3
    assert tqdm.max == 3
    assert tqdm.text.startswith('100% 3/3')
    assert isinstance(tqdm.progress, pn.widgets.indicators.Progress)
    assert isinstance(tqdm.text_pane, pn.pane.Str)
    assert isinstance(tqdm.layout, pn.Row)