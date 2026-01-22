import traceback
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
Looks up "name".

    Args:
      name: a string specifying the registry key for the candidate.
    Returns:
      Registered object if found
    Raises:
      LookupError: if "name" has not been registered.
    