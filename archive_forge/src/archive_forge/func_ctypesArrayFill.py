import os
import platform
import shutil
import sys
import ctypes
from time import monotonic as clock
import configparser
from typing import Union
from .. import sparse
from .. import constants as const
import logging
import subprocess
from uuid import uuid4
def ctypesArrayFill(myList, type=ctypes.c_double):
    """
    Creates a c array with ctypes from a python list
    type is the type of the c array
    """
    ctype = type * len(myList)
    cList = ctype()
    for i, elem in enumerate(myList):
        cList[i] = elem
    return cList