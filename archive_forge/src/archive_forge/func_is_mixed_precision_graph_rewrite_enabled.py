from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.train.is_mixed_precision_graph_rewrite_enabled', v1=[])
def is_mixed_precision_graph_rewrite_enabled():
    return _mixed_precision_graph_rewrite_is_enabled