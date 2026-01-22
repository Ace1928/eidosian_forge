from distutils.cmd import Command
from distutils import log, dir_util
import os, sys, re
Convert a project or version name to its filename-escaped form

    Any '-' characters are currently replaced with '_'.
    