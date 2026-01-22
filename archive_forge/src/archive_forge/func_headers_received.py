from collections import deque
from io import BytesIO
from ... import debug, errors
from ...trace import mutter
def headers_received(self, headers):
    """Called when message headers are received.

        This default implementation just stores them in self.headers.
        """
    self.headers = headers