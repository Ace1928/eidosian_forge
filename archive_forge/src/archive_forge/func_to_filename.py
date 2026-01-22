from distutils.cmd import Command
from distutils import log, dir_util
import os, sys, re
def to_filename(name):
    """Convert a project or version name to its filename-escaped form

    Any '-' characters are currently replaced with '_'.
    """
    return name.replace('-', '_')