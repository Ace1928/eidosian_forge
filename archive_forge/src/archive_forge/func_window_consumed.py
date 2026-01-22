from __future__ import division
from .exceptions import FlowControlError
def window_consumed(self, size):
    """
        We have received a certain number of bytes from the remote peer. This
        necessarily shrinks the flow control window!

        :param size: The number of flow controlled bytes we received from the
            remote peer.
        :type size: ``int``
        :returns: Nothing.
        :rtype: ``None``
        """
    self.current_window_size -= size
    if self.current_window_size < 0:
        raise FlowControlError('Flow control window shrunk below 0')