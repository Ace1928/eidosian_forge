import param
import pytest
from panel.layout import Column, Row
from panel.pane import HoloViews
from panel.param import ParamMethod
from panel.pipeline import Pipeline, find_route
from panel.widgets import Button, Select
def test_pipeline_error_condition():
    pipeline = Pipeline()
    stage2b = Stage2b(root='error')
    pipeline.add_stage('Stage 2', Stage2)
    pipeline.add_stage('Stage 2b', stage2b)
    pipeline.add_stage('Stage 1', Stage1)
    pipeline.define_graph({'Stage 1': ('Stage 2', 'Stage 2b')})
    pipeline.next_selector.value = 'Stage 2b'
    pipeline._next()
    assert isinstance(pipeline.error[0], Button)
    stage2b.root = 2
    pipeline._next()
    assert len(pipeline.error) == 0