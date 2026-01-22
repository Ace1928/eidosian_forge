import time
import numpy as np
import pandas as pd
import pytest
from tqdm.contrib.concurrent import process_map
import panel as pn
from panel.widgets import Tqdm
def get_tqdm_app():
    tqdm = Tqdm(layout='row', sizing_mode='stretch_width')

    def run(*events):
        for _ in tqdm(range(10)):
            time.sleep(0.2)
    button = pn.widgets.Button(name='Run Loop', button_type='primary')
    button.on_click(run)
    tqdm.tqdm.pandas(desc='my bar!')
    df = pd.DataFrame(np.random.randint(0, 100, (100000, 6)))

    def run_df(*events):
        df.progress_apply(lambda x: x ** 2)
    pandas_button = pn.widgets.Button(name='Pandas Apply', button_type='success')
    pandas_button.on_click(run_df)
    component = pn.Column(button, pandas_button, tqdm, sizing_mode='stretch_width')
    template = pn.template.FastListTemplate(title='Panel - Tqdm Indicator', main=[component], sidebar=[pn.Param(tqdm, sizing_mode='stretch_width')])
    return template