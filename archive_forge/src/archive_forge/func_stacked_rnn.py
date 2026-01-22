import collections
import keras.src as keras
def stacked_rnn():
    """Stacked RNN model."""
    inputs = keras.Input((None, 3))
    layer = keras.layers.RNN([keras.layers.LSTMCell(2) for _ in range(3)])
    x = layer(inputs)
    outputs = keras.layers.Dense(2)(x)
    model = keras.Model(inputs, outputs)
    return ModelFn(model, (None, 4, 3), (None, 2))