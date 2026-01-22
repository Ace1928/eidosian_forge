from keras.src.applications import imagenet_utils
from keras.src.applications import resnet
from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.applications.resnet_v2.decode_predictions')
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)