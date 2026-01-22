import contextlib
from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.tpu import tpu_replication
def enclosing_tpu_context_and_graph():
    """Returns the TPUReplicateContext which exists inside a tpu.rewrite(), and its associated graph."""
    graph = ops.get_default_graph()
    while graph is not None:
        ctx = graph._get_control_flow_context()
        while ctx is not None:
            if isinstance(ctx, tpu_replication.TPUReplicateContext):
                return (ctx, graph)
            ctx = ctx.outer_context
        graph = getattr(graph, 'outer_graph', None)
    return (None, None)