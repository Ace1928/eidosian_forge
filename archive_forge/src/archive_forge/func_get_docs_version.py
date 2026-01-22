from __future__ import unicode_literals
import datetime
import os
import subprocess
def get_docs_version(version=None):
    version = get_complete_version(version)
    if version[3] != 'final':
        return 'dev'
    else:
        return '%d.%d' % version[:2]