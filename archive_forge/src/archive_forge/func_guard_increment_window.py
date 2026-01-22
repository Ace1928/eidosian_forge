import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def guard_increment_window(current, increment):
    """
    Increments a flow control window, guarding against that window becoming too
    large.

    :param current: The current value of the flow control window.
    :param increment: The increment to apply to that window.
    :returns: The new value of the window.
    :raises: ``FlowControlError``
    """
    LARGEST_FLOW_CONTROL_WINDOW = 2 ** 31 - 1
    new_size = current + increment
    if new_size > LARGEST_FLOW_CONTROL_WINDOW:
        raise FlowControlError('May not increment flow control window past %d' % LARGEST_FLOW_CONTROL_WINDOW)
    return new_size