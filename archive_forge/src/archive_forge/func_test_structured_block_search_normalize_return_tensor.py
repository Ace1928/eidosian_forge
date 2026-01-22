import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import analysers
from autokeras import blocks
from autokeras import test_utils
def test_structured_block_search_normalize_return_tensor():
    block = blocks.StructuredDataBlock(name='a')
    block.column_names = ['0', '1']
    block.column_types = {'0': analysers.NUMERICAL, '1': analysers.NUMERICAL}
    hp = keras_tuner.HyperParameters()
    hp.values['a/' + blocks.wrapper.NORMALIZE] = True
    outputs = block.build(hp, keras.Input(shape=(2,), dtype=tf.string))
    assert len(nest.flatten(outputs)) == 1