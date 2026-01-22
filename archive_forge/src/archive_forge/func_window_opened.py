from __future__ import division
from .exceptions import FlowControlError
def window_opened(self, size):
    """
        The flow control window has been incremented, either because of manual
        flow control management or because of the user changing the flow
        control settings. This can have the effect of increasing what we
        consider to be the "maximum" flow control window size.

        This does not increase our view of how many bytes have been processed,
        only of how much space is in the window.

        :param size: The increment to the flow control window we received.
        :type size: ``int``
        :returns: Nothing
        :rtype: ``None``
        """
    self.current_window_size += size
    if self.current_window_size > LARGEST_FLOW_CONTROL_WINDOW:
        raise FlowControlError("Flow control window mustn't exceed %d" % LARGEST_FLOW_CONTROL_WINDOW)
    if self.current_window_size > self.max_window_size:
        self.max_window_size = self.current_window_size