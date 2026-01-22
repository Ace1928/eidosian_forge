import argparse
import getpass
import inspect
import io
import lzma
import os
import pathlib
import platform
import re
import shutil
import sys
from lzma import CHECK_CRC64, CHECK_SHA256, is_check_supported
from typing import Any, List, Optional
import _lzma  # type: ignore
import multivolumefile
import texttable  # type: ignore
import py7zr
from py7zr.callbacks import ExtractCallback
from py7zr.compressor import SupportedMethods
from py7zr.helpers import Local
from py7zr.properties import COMMAND_HELP_STRING
@staticmethod
def print_archiveinfo(archive, file):
    file.write('--\n')
    file.write('Path = {}\n'.format(archive.filename))
    file.write('Type = 7z\n')
    fstat = os.stat(archive.filename)
    file.write('Phisical Size = {}\n'.format(fstat.st_size))
    file.write('Headers Size = {}\n'.format(archive.header.size))
    file.write('Method = {}\n'.format(', '.join(archive._get_method_names())))
    if archive._is_solid():
        file.write('Solid = {}\n'.format('+'))
    else:
        file.write('Solid = {}\n'.format('-'))
    file.write('Blocks = {}\n'.format(len(archive.header.main_streams.unpackinfo.folders)))