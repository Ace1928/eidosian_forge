import param
import pytest
from panel.layout import Column, Row
from panel.pane import HoloViews
from panel.param import ParamMethod
from panel.pipeline import Pipeline, find_route
from panel.widgets import Button, Select
def test_pipeline_from_instances():
    pipeline = Pipeline([('Stage 1', Stage1()), ('Stage 2', Stage2())])
    layout = pipeline.layout
    assert isinstance(layout, Column)
    assert isinstance(layout[0], Row)
    (title, error), progress, (prev_button, next_button) = layout[0].objects
    assert isinstance(error, Row)
    assert isinstance(prev_button, Button)
    assert isinstance(next_button, Button)
    assert isinstance(progress, HoloViews)
    hv_obj = progress.object
    graph = hv_obj.get(0)
    assert isinstance(graph, hv.Graph)
    assert len(graph) == 1
    labels = hv_obj.get(1)
    assert isinstance(labels, hv.Labels)
    assert list(labels['Stage']) == ['Stage 1', 'Stage 2']
    stage = layout[1][0]
    assert isinstance(stage, Row)
    assert isinstance(stage[1], ParamMethod)
    assert stage[1].object() == '5 * 5 = 25'
    pipeline.param.trigger('next')
    stage = layout[1][0]
    assert isinstance(stage, Row)
    assert isinstance(stage[1], ParamMethod)
    assert stage[1].object() == '25^0.1=1.380'
    pipeline.param.trigger('previous')
    stage = layout[1][0]
    assert isinstance(stage, Row)
    assert isinstance(stage[1], ParamMethod)
    assert stage[1].object() == '5 * 5 = 25'