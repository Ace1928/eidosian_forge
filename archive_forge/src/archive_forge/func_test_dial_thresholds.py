import pytest
from panel.widgets.indicators import (
def test_dial_thresholds(document, comm):
    dial = Dial(value=0, colors=[(0.33, 'green'), (0.66, 'yellow'), (1, 'red')])
    model = dial.get_root(document, comm)
    cds = model.select(name='annulus_source')
    assert ['green', 'whitesmoke'] == cds.data['color']
    dial.value = 50
    assert ['yellow', 'whitesmoke'] == cds.data['color']
    dial.value = 72
    assert ['red', 'whitesmoke'] == cds.data['color']