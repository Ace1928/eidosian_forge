import os
import keras_tuner
import pytest
import autokeras as ak
from autokeras import graph as graph_module
def test_hyper_graph_cycle():
    input_node1 = ak.Input()
    input_node2 = ak.Input()
    output_node1 = ak.DenseBlock()(input_node1)
    output_node2 = ak.DenseBlock()(input_node2)
    output_node = ak.Merge()([output_node1, output_node2])
    head = ak.RegressionHead()
    output_node = head(output_node)
    head.outputs = output_node1
    with pytest.raises(ValueError) as info:
        graph_module.Graph(inputs=[input_node1, input_node2], outputs=output_node)
    assert 'The network has a cycle.' in str(info.value)