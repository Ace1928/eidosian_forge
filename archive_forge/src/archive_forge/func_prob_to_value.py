from keras_tuner.src import protos
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import conditions as conditions_mod
from keras_tuner.src.engine.hyperparameters import hp_utils
from keras_tuner.src.engine.hyperparameters.hp_types import numerical
def prob_to_value(self, prob):
    if self.step is None:
        return int(self._sample_numerical_value(prob, self.max_value + 1))
    return int(self._sample_with_step(prob))