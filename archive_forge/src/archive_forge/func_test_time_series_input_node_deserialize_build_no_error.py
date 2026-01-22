import keras_tuner
from autokeras import blocks
from autokeras import nodes
def test_time_series_input_node_deserialize_build_no_error():
    node = nodes.TimeseriesInput(lookback=2, shape=(32,))
    node = nodes.deserialize(nodes.serialize(node))
    hp = keras_tuner.HyperParameters()
    input_node = node.build_node(hp)
    node.build(hp, input_node)