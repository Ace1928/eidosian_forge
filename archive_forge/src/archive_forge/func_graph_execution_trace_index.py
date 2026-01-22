import numpy as np
from tensorflow.core.protobuf import debug_event_pb2
@property
def graph_execution_trace_index(self):
    return self._graph_execution_trace_index