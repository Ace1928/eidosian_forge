from tensorflow.keras import layers, models
from tensorflow.keras.datasets import boston_housing
from tune_tensorflow import KerasTrainingSpec
def get_fit_metric(self, history):
    return float(history.history['val_mae'][-1])