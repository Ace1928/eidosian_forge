from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def get_address_space(self):
    return conf.lib.clang_getAddressSpace(self)