import sys, os, pickle
from hashlib import md5
from xml.sax.saxutils import quoteattr
from time import process_time as clock
from reportlab.lib.utils import asBytes, asNative as _asNative
from reportlab.lib.utils import rl_isdir, rl_isfile, rl_listdir, rl_getmtime
def getTag(self):
    """Return an XML tag representation"""
    attrs = []
    for k, v in self.__dict__.items():
        if k not in ['timeModified']:
            if v:
                attrs.append('%s=%s' % (k, quoteattr(str(v))))
    return '<font ' + ' '.join(attrs) + '/>'