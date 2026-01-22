from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.register_call_context_function', v1=[])
def register_call_context_function(func):
    global _KERAS_CALL_CONTEXT_FUNCTION
    _KERAS_CALL_CONTEXT_FUNCTION = func