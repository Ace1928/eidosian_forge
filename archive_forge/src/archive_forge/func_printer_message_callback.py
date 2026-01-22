import sys
import threading
from typing import Callable, List, Sequence, TextIO
from absl import logging
def printer_message_callback(*, file: TextIO=sys.stdout, prefix: str='') -> SolveMessageCallback:
    """Returns a message callback that prints to a file.

    It prints its output to the given text file, prefixing each line with the
    given prefix.

    For each call to the returned message callback, the output_stream is flushed.

    Args:
      file: The file to print to. It prints to stdout by default.
      prefix: The prefix to print in front of each line.

    Returns:
      A function matching the expected signature for message callbacks.
    """
    mutex = threading.Lock()

    def callback(messages: Sequence[str]) -> None:
        with mutex:
            for message in messages:
                file.write(prefix)
                file.write(message)
                file.write('\n')
            file.flush()
    return callback