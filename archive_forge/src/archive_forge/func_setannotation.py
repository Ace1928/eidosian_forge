import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def setannotation(self, *args):
    """(typ, [data]) = <instance>.setannotation(mailbox[, entry, attribute]+)
        Set ANNOTATIONs."""
    typ, dat = self._simple_command('SETANNOTATION', *args)
    return self._untagged_response(typ, dat, 'ANNOTATION')