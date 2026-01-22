import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def proxyauth(self, user):
    """Assume authentication as "user".

        Allows an authorised administrator to proxy into any user's
        mailbox.

        (typ, [data]) = <instance>.proxyauth(user)
        """
    name = 'PROXYAUTH'
    return self._simple_command('PROXYAUTH', user)