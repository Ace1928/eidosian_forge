from io import StringIO, BytesIO
import codecs
import os
import sys
import re
import errno
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .expect import Expecter, searcher_string, searcher_re
def read_nonblocking(self, size=1, timeout=None):
    """This reads data from the file descriptor.

        This is a simple implementation suitable for a regular file. Subclasses using ptys or pipes should override it.

        The timeout parameter is ignored.
        """
    try:
        s = os.read(self.child_fd, size)
    except OSError as err:
        if err.args[0] == errno.EIO:
            self.flag_eof = True
            raise EOF('End Of File (EOF). Exception style platform.')
        raise
    if s == b'':
        self.flag_eof = True
        raise EOF('End Of File (EOF). Empty string style platform.')
    s = self._decoder.decode(s, final=False)
    self._log(s, 'read')
    return s