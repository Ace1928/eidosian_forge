import collections
import logging
import os
import re
import socket
import sys
from humanfriendly import coerce_boolean
from humanfriendly.compat import coerce_string, is_string, on_windows
from humanfriendly.terminal import ANSI_COLOR_CODES, ansi_wrap, enable_ansi_support, terminal_supports_colors
from humanfriendly.text import format, split
def replace_handler(logger, match_handler, reconfigure):
    """
    Prepare to replace a handler.

    :param logger: Refer to :func:`find_handler()`.
    :param match_handler: Refer to :func:`find_handler()`.
    :param reconfigure: :data:`True` if an existing handler should be replaced,
                        :data:`False` otherwise.
    :returns: A tuple of two values:

              1. The matched :class:`~logging.Handler` object or :data:`None`
                 if no handler was matched.
              2. The :class:`~logging.Logger` to which the matched handler was
                 attached or the logger given to :func:`replace_handler()`.
    """
    handler, other_logger = find_handler(logger, match_handler)
    if handler and other_logger and reconfigure:
        other_logger.removeHandler(handler)
        logger = other_logger
    return (handler, logger)