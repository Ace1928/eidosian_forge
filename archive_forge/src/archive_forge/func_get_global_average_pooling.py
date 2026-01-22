from tensorflow.keras import layers
def get_global_average_pooling(shape):
    return [layers.GlobalAveragePooling1D, layers.GlobalAveragePooling2D, layers.GlobalAveragePooling3D][len(shape) - 3]