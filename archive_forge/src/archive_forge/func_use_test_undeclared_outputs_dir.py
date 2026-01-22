import os
import os.path
import re
from absl import flags
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
def use_test_undeclared_outputs_dir(self):
    """Decides the output directory of the report and trace files.

    Args:
       None.

    Returns:
       True if the output files should be written to the
       test-undeclared-outputs-directory defined via an
       env variable.
    """
    return self.is_flag_on(FLAG_NAME_USE_TEST_UNDECLARED_OUTPUTS_DIR)