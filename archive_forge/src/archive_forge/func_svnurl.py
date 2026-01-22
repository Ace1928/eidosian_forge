import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def svnurl(self):
    """ return current SvnPath for this WC-item. """
    info = self.info()
    return py.path.svnurl(info.url)