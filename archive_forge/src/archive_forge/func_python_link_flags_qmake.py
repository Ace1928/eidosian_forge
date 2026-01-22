import os, glob, re, sys
from distutils import sysconfig
def python_link_flags_qmake():
    flags = python_link_data()
    if sys.platform == 'win32':
        libdir = flags['libdir']
        for d in libdir.split('\\'):
            if ' ' in d:
                libdir = libdir.replace(d, d.split(' ')[0][:-1] + '~1')
        return '-L{} -l{}'.format(libdir, flags['lib'])
    elif sys.platform == 'darwin':
        return '-L{} -l{}'.format(flags['libdir'], flags['lib'])
    else:
        return '-L{} -l{}'.format(flags['libdir'], flags['lib'])