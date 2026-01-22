from typing import List
from jupyter_client.channelsabc import HBChannelABC
def is_beating(self):
    """Test if the channel is beating."""
    return not self._pause