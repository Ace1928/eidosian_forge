from keras.src import backend
from keras.src import ops
def get_dropout_mask(self, step_input):
    if not hasattr(self, '_dropout_mask'):
        self._dropout_mask = None
    if self._dropout_mask is None and self.dropout > 0:
        ones = ops.ones_like(step_input)
        self._dropout_mask = backend.random.dropout(ones, rate=self.dropout, seed=self.seed_generator)
    return self._dropout_mask