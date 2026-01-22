import readline
from tensorflow.python.debug.cli import base_ui
from tensorflow.python.debug.cli import debugger_cli_common
Dispatch user command.

    Args:
      command: (str) Command to dispatch.

    Returns:
      An exit token object. None value means that the UI loop should not exit.
      A non-None value means the UI loop should exit.
    