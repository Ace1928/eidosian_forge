import pytest
from panel.widgets.indicators import (
def test_tqdm_color():
    tqdm = Tqdm()
    tqdm.text_pane.styles = {'color': 'green'}
    for _ in tqdm(range(2)):
        pass
    assert tqdm.text_pane.styles['color'] == 'green'