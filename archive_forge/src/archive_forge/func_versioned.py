import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def versioned(self):
    try:
        s = self.svnwcpath.info()
    except (py.error.ENOENT, py.error.EEXIST):
        return False
    except py.process.cmdexec.Error:
        e = sys.exc_info()[1]
        if e.err.find('is not a working copy') != -1:
            return False
        if e.err.lower().find('not a versioned resource') != -1:
            return False
        raise
    else:
        return True