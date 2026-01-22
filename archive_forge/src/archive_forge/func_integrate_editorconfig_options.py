from __future__ import print_function
import sys
import os
import platform
import io
import getopt
import re
import string
import errno
import copy
import glob
from jsbeautifier.__version__ import __version__
from jsbeautifier.javascript.options import BeautifierOptions
from jsbeautifier.javascript.beautifier import Beautifier
def integrate_editorconfig_options(filepath, local_options, outfile, default_file_type):
    if getattr(local_options, 'editorconfig'):
        editorconfig_filepath = filepath
        if editorconfig_filepath == '-':
            if outfile != 'stdout':
                editorconfig_filepath = outfile
            else:
                fileType = default_file_type
                editorconfig_filepath = 'stdin.' + fileType
        local_options = copy.copy(local_options)
        set_file_editorconfig_opts(editorconfig_filepath, local_options)
    return local_options