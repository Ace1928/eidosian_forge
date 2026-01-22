import os
import keras_tuner
import pytest
import autokeras as ak
from autokeras import graph as graph_module
def test_graph_basics():
    input_node = ak.Input(shape=(30,))
    output_node = input_node
    output_node = ak.DenseBlock()(output_node)
    output_node = ak.RegressionHead(shape=(1,))(output_node)
    model = graph_module.Graph(inputs=input_node, outputs=output_node).build(keras_tuner.HyperParameters())
    assert model.input_shape == (None, 30)
    assert model.output_shape == (None, 1)