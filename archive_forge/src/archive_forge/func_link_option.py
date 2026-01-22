import os, glob, re, sys
from distutils import sysconfig
def link_option(lib):
    baseName = os.path.basename(lib)
    link = ' -l'
    if sys.platform in ['linux', 'linux2']:
        link = lib
    elif sys.platform in ['darwin']:
        link += os.path.splitext(baseName[3:])[0]
    else:
        link += os.path.splitext(baseName)[0]
    return link