import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
def login_cram_md5(self, user, password):
    """ Force use of CRAM-MD5 authentication.

        (typ, [data]) = <instance>.login_cram_md5(user, password)
        """
    self.user, self.password = (user, password)
    return self.authenticate('CRAM-MD5', self._CRAM_MD5_AUTH)