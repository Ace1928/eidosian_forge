from __future__ import absolute_import, unicode_literals
import re
import sys
def normalize_uri(uri):
    try:
        return quote(uri.encode('utf-8'), safe=str('/@:+?=&()%#*,'))
    except UnicodeDecodeError:
        s = quote(uri.encode('utf-8'))
        s = re.sub('%40', '@', s)
        s = re.sub('%3A', ':', s)
        s = re.sub('%2B', '+', s)
        s = re.sub('%3F', '?', s)
        s = re.sub('%3D', '=', s)
        s = re.sub('%26', '&', s)
        s = re.sub('%28', '(', s)
        s = re.sub('%29', ')', s)
        s = re.sub('%25', '%', s)
        s = re.sub('%23', '#', s)
        s = re.sub('%2A', '*', s)
        s = re.sub('%2C', ',', s)
        return s