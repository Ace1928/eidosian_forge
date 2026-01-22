import numpy as np
from tensorflow.core.protobuf import debug_event_pb2
@property
def execution_index(self):
    return self._execution_index