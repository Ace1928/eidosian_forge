import threading
from paramiko import util
from paramiko.common import (
def get_banner(self):
    """
        A pre-login banner to display to the user. The message may span
        multiple lines separated by crlf pairs. The language should be in
        rfc3066 style, for example: en-US

        The default implementation always returns ``(None, None)``.

        :returns: A tuple containing the banner and language code.

        .. versionadded:: 2.3
        """
    return (None, None)