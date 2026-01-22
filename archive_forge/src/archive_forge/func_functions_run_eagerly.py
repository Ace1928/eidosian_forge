from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export('config.functions_run_eagerly')
def functions_run_eagerly():
    """Returns the value of the `run_functions_eagerly` setting."""
    return RUN_FUNCTIONS_EAGERLY