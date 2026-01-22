import os, glob, re, sys
from distutils import sysconfig
def find_shiboken2_module():
    return find_package_path('shiboken2')