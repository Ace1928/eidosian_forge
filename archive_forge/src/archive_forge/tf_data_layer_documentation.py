import tree
import keras.src.backend
from keras.src.layers.layer import Layer
from keras.src.random.seed_generator import SeedGenerator
from keras.src.utils import backend_utils
from keras.src.utils import tracking
Layer that can safely used in a tf.data pipeline.

    The `call()` method must solely rely on `self.backend` ops.

    Only supports a single input tensor argument.
    