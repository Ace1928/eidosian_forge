from tensorflow.core.grappler.costs import op_performance_data_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.grappler import _pywrap_tf_item as tf_item
@property
def tf_item(self):
    if self._item_graph != self._metagraph:
        self._BuildTFItem()
        self._item_graph.CopyFrom(self._metagraph)
    return self._tf_item