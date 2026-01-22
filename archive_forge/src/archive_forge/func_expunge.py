import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def expunge(self):
    """Permanently remove deleted items from selected mailbox.

        Generates 'EXPUNGE' response for each deleted message.

        (typ, [data]) = <instance>.expunge()

        'data' is list of 'EXPUNGE'd message numbers in order received.
        """
    name = 'EXPUNGE'
    typ, dat = self._simple_command(name)
    return self._untagged_response(typ, dat, name)