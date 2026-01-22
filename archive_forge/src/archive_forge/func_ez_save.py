import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def ez_save(self, filename, garbage=3, clean=False, deflate=True, deflate_images=True, deflate_fonts=True, incremental=False, ascii=False, expand=False, linear=False, pretty=False, encryption=1, permissions=4095, owner_pw=None, user_pw=None, no_new_id=True, preserve_metadata=1, use_objstms=1, compression_effort=0):
    """
        Save PDF using some different defaults
        """
    return self.save(filename, garbage=garbage, clean=clean, deflate=deflate, deflate_images=deflate_images, deflate_fonts=deflate_fonts, incremental=incremental, ascii=ascii, expand=expand, linear=linear, pretty=pretty, encryption=encryption, permissions=permissions, owner_pw=owner_pw, user_pw=user_pw, no_new_id=no_new_id, preserve_metadata=preserve_metadata, use_objstms=use_objstms, compression_effort=compression_effort)