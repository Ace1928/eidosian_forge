import sys
import threading
from typing import Callable, List, Sequence, TextIO
from absl import logging
def log_messages(messages: Sequence[str], *, level: int=logging.INFO, prefix: str='') -> None:
    """Logs the input messages from a message callback using absl.logging.log().

    It logs each line with the given prefix. It setups absl.logging so that the
    logs use the file name and line of the caller of this function.

      Typical usage example:

      result = solve.solve(
          model, parameters.SolverType.GSCIP,
          msg_cb=lambda msgs: message_callback.log_messages(
              msgs, prefix='[solver] '))

    Args:
      messages: The messages received in the message callback (typically a lambda
        function in the caller code).
      level: One of absl.logging.(DEBUG|INFO|WARNING|ERROR|FATAL).
      prefix: The prefix to print in front of each line.
    """
    for message in messages:
        logging.log(level, '%s%s', prefix, message)