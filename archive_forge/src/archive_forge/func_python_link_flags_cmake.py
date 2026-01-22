import os, glob, re, sys
from distutils import sysconfig
def python_link_flags_cmake():
    flags = python_link_data()
    libdir = flags['libdir']
    lib = re.sub('.dll$', '.lib', flags['lib'])
    return '{};{}'.format(libdir, lib)