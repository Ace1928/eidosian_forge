import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def makecmdoptions(self):
    uname = self.username.replace('"', '\\"')
    passwd = self.password.replace('"', '\\"')
    ret = []
    if uname:
        ret.append('--username="%s"' % (uname,))
    if passwd:
        ret.append('--password="%s"' % (passwd,))
    if not self.cache_auth:
        ret.append('--no-auth-cache')
    if not self.interactive:
        ret.append('--non-interactive')
    return ' '.join(ret)