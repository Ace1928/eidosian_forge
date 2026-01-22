import collections
import gast
from tensorflow.python.autograph.pyct import gast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
def new_name(self, stem='tmp'):
    self._idx += 1
    return stem + '_' + str(1000 + self._idx)