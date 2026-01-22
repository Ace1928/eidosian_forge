import collections
import keras.src as keras
def nested_subclassed_model():
    """A subclass model nested in another subclass model."""

    class NestedSubclassModel(keras.Model):
        """A nested subclass model."""

        def __init__(self):
            super().__init__()
            self.dense1 = keras.layers.Dense(4, activation='relu')
            self.dense2 = keras.layers.Dense(2, activation='relu')
            self.bn = keras.layers.BatchNormalization()
            self.inner_subclass_model = MySubclassModel()

        def call(self, inputs):
            x = self.dense1(inputs)
            x = self.bn(x)
            x = self.inner_subclass_model(x)
            return self.dense2(x)
    return ModelFn(NestedSubclassModel(), (None, 3), (None, 2))