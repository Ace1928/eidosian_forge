import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def getquotaroot(self, mailbox):
    """Get the list of quota roots for the named mailbox.

        (typ, [[QUOTAROOT responses...], [QUOTA responses]]) = <instance>.getquotaroot(mailbox)
        """
    typ, dat = self._simple_command('GETQUOTAROOT', mailbox)
    typ, quota = self._untagged_response(typ, dat, 'QUOTA')
    typ, quotaroot = self._untagged_response(typ, dat, 'QUOTAROOT')
    return (typ, [quotaroot, quota])