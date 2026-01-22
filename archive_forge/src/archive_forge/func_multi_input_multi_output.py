import collections
import keras.src as keras
def multi_input_multi_output():
    """Multi-input Multi-output model."""
    body_input = keras.Input(shape=(None,), name='body')
    tags_input = keras.Input(shape=(2,), name='tags')
    x = keras.layers.Embedding(10, 4)(body_input)
    body_features = keras.layers.LSTM(5)(x)
    x = keras.layers.concatenate([body_features, tags_input])
    pred_1 = keras.layers.Dense(2, activation='sigmoid', name='priority')(x)
    pred_2 = keras.layers.Dense(3, activation='softmax', name='department')(x)
    model = keras.Model(inputs=[body_input, tags_input], outputs=[pred_1, pred_2])
    return ModelFn(model, [(None, 1), (None, 2)], [(None, 2), (None, 3)])