from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(None, 'Use tf.config.functions_run_eagerly instead of the experimental version.')
@tf_export('config.experimental_functions_run_eagerly')
def experimental_functions_run_eagerly():
    """Returns the value of the `experimental_run_functions_eagerly` setting."""
    return functions_run_eagerly()