from tensorflow.keras import layers, models
from tensorflow.keras.datasets import boston_housing
from tune_tensorflow import KerasTrainingSpec
def get_fit_params(self):
    return ([self.train_data, self.train_targets], dict(validation_data=(self.test_data, self.test_targets), shuffle=True))