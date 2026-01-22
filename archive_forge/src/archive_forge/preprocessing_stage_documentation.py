import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.engine import base_preprocessing_layer
from keras.src.engine import functional
from keras.src.engine import sequential
from keras.src.utils import tf_utils
Maps this object's inputs to those at current_layer_index.

                Args:
                  x: Batch of inputs seen in entry of the `PreprocessingStage`
                    instance.

                Returns:
                  Batch of inputs to be processed by layer
                    `self.layers[current_layer_index]`
                