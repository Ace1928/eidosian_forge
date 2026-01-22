from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.train.set_using_mixed_precision_policy', v1=[])
def set_using_mixed_precision_policy(is_using):
    global _using_mixed_precision_policy
    _using_mixed_precision_policy = is_using