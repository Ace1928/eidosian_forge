import param
import pytest
from panel.layout import Column, Row
from panel.pane import HoloViews
from panel.param import ParamMethod
from panel.pipeline import Pipeline, find_route
from panel.widgets import Button, Select
def test_pipeline_define_graph():
    pipeline = Pipeline()
    pipeline.add_stage('Stage 2', Stage2)
    pipeline.add_stage('Stage 2b', Stage2b)
    pipeline.add_stage('Stage 1', Stage1)
    pipeline.define_graph({'Stage 1': ('Stage 2', 'Stage 2b')})
    assert pipeline._stage == 'Stage 1'
    assert isinstance(pipeline.buttons, Row)
    (pselect, pbutton), (nselect, nbutton) = pipeline.buttons
    assert isinstance(pselect, Select)
    assert pselect.disabled
    assert isinstance(pbutton, Button)
    assert pbutton.disabled
    assert isinstance(nselect, Select)
    assert not nselect.disabled
    assert nselect.options == ['Stage 2', 'Stage 2b']
    assert nselect.value == 'Stage 2'
    assert isinstance(nbutton, Button)
    assert not nbutton.disabled
    pipeline._next()
    assert isinstance(pipeline._state, Stage2)
    pipeline._previous()
    nselect.value = 'Stage 2b'
    pipeline._next()
    assert isinstance(pipeline._state, Stage2b)