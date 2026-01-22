import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def getacl(self, mailbox):
    """Get the ACLs for a mailbox.

        (typ, [data]) = <instance>.getacl(mailbox)
        """
    typ, dat = self._simple_command('GETACL', mailbox)
    return self._untagged_response(typ, dat, 'ACL')