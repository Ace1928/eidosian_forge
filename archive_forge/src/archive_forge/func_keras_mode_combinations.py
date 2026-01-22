import functools
from tensorflow.python import tf2
from tensorflow.python.framework import combinations
from tensorflow.python.framework import test_combinations
from tensorflow.python.keras import testing_utils
def keras_mode_combinations(mode=None, run_eagerly=None):
    """Returns the default test combinations for tf.keras tests.

  Note that if tf2 is enabled, then v1 session test will be skipped.

  Args:
    mode: List of modes to run the tests. The valid options are 'graph' and
      'eager'. Default to ['graph', 'eager'] if not specified. If a empty list
      is provide, then the test will run under the context based on tf's
      version, eg graph for v1 and eager for v2.
    run_eagerly: List of `run_eagerly` value to be run with the tests.
      Default to [True, False] if not specified. Note that for `graph` mode,
      run_eagerly value will only be False.

  Returns:
    A list contains all the combinations to be used to generate test cases.
  """
    if mode is None:
        mode = ['eager'] if tf2.enabled() else ['graph', 'eager']
    if run_eagerly is None:
        run_eagerly = [True, False]
    result = []
    if 'eager' in mode:
        result += combinations.combine(mode=['eager'], run_eagerly=run_eagerly)
    if 'graph' in mode:
        result += combinations.combine(mode=['graph'], run_eagerly=[False])
    return result