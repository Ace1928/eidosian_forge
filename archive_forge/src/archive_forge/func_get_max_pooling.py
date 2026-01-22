from tensorflow.keras import layers
def get_max_pooling(shape):
    return [layers.MaxPool1D, layers.MaxPool2D, layers.MaxPool3D][len(shape) - 3]