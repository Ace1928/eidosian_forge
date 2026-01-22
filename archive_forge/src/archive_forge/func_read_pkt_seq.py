from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def read_pkt_seq(self):
    """Read a sequence of pkt-lines from the remote git process.

        Returns: Yields each line of data up to but not including the next
            flush-pkt.
        """
    pkt = self.read_pkt_line()
    while pkt:
        yield pkt
        pkt = self.read_pkt_line()