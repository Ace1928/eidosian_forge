import logging as _logging
import os as _os
import sys as _sys
import _thread
import time as _time
import traceback as _traceback
from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import WARN
import threading
from tensorflow.python.util.tf_export import tf_export
@tf_export('get_logger')
def get_logger():
    """Return TF logger instance.

  Returns:
    An instance of the Python logging library Logger.

  See Python documentation (https://docs.python.org/3/library/logging.html)
  for detailed API. Below is only a summary.

  The logger has 5 levels of logging from the most serious to the least:

  1. FATAL
  2. ERROR
  3. WARN
  4. INFO
  5. DEBUG

  The logger has the following methods, based on these logging levels:

  1. fatal(msg, *args, **kwargs)
  2. error(msg, *args, **kwargs)
  3. warn(msg, *args, **kwargs)
  4. info(msg, *args, **kwargs)
  5. debug(msg, *args, **kwargs)

  The `msg` can contain string formatting.  An example of logging at the `ERROR`
  level
  using string formating is:

  >>> tf.get_logger().error("The value %d is invalid.", 3)

  You can also specify the logging verbosity.  In this case, the
  WARN level log will not be emitted:

  >>> tf.get_logger().setLevel(ERROR)
  >>> tf.get_logger().warn("This is a warning.")
  """
    global _logger
    if _logger:
        return _logger
    _logger_lock.acquire()
    try:
        if _logger:
            return _logger
        logger = _logging.getLogger('tensorflow')
        logger.findCaller = _logger_find_caller
        if not _logging.getLogger().handlers:
            _interactive = False
            try:
                if _sys.ps1:
                    _interactive = True
            except AttributeError:
                _interactive = _sys.flags.interactive
            if _interactive:
                logger.setLevel(INFO)
                _logging_target = _sys.stdout
            else:
                _logging_target = _sys.stderr
            _handler = _logging.StreamHandler(_logging_target)
            _handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT, None))
            logger.addHandler(_handler)
        _logger = logger
        return _logger
    finally:
        _logger_lock.release()