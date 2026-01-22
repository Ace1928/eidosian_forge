import threading
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
def get_default(self):
    return self.stack[-1] if self.stack else None