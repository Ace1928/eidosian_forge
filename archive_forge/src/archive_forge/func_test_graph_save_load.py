import os
import keras_tuner
import pytest
import autokeras as ak
from autokeras import graph as graph_module
def test_graph_save_load(tmp_path):
    input1 = ak.Input()
    input2 = ak.Input()
    output1 = ak.DenseBlock()(input1)
    output2 = ak.ConvBlock()(input2)
    output = ak.Merge()([output1, output2])
    output1 = ak.RegressionHead()(output)
    output2 = ak.ClassificationHead()(output)
    graph = graph_module.Graph(inputs=[input1, input2], outputs=[output1, output2])
    path = os.path.join(tmp_path, 'graph')
    graph.save(path)
    graph = graph_module.load_graph(path)
    assert len(graph.inputs) == 2
    assert len(graph.outputs) == 2
    assert isinstance(graph.inputs[0].out_blocks[0], ak.DenseBlock)
    assert isinstance(graph.inputs[1].out_blocks[0], ak.ConvBlock)