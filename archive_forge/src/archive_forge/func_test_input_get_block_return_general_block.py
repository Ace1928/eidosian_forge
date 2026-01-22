import keras_tuner
from autokeras import blocks
from autokeras import nodes
def test_input_get_block_return_general_block():
    input_node = nodes.Input()
    assert isinstance(input_node.get_block(), blocks.GeneralBlock)