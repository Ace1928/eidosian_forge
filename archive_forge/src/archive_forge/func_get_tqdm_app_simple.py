import time
import numpy as np
import pandas as pd
import pytest
from tqdm.contrib.concurrent import process_map
import panel as pn
from panel.widgets import Tqdm
def get_tqdm_app_simple():
    import time
    tqdm = Tqdm(layout='row', sizing_mode='stretch_width')

    def run(*events):
        for _ in tqdm(range(10)):
            time.sleep(0.2)
    button = pn.widgets.Button(name='Run Loop', button_type='primary')
    button.on_click(run)
    return pn.Column(tqdm, button)