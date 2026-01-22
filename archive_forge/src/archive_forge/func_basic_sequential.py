import collections
import keras.src as keras
def basic_sequential():
    """Basic sequential model."""
    model = keras.Sequential([keras.layers.Dense(3, activation='relu', input_shape=(3,)), keras.layers.Dense(2, activation='softmax')])
    return ModelFn(model, (None, 3), (None, 2))