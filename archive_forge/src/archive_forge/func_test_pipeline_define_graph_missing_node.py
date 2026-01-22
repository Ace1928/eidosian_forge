import param
import pytest
from panel.layout import Column, Row
from panel.pane import HoloViews
from panel.param import ParamMethod
from panel.pipeline import Pipeline, find_route
from panel.widgets import Button, Select
def test_pipeline_define_graph_missing_node():
    pipeline = Pipeline()
    pipeline.add_stage('Stage 1', Stage1)
    pipeline.add_stage('Stage 2', Stage2)
    with pytest.raises(ValueError):
        pipeline.define_graph({'Stage 1': ('Stage 2', 'Stage 2b')})