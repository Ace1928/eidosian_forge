import sys
import shutil
from getopt import getopt, GetoptError
import os
from os import environ, mkdir
from os.path import dirname, join, basename, exists, expanduser
import pkgutil
import re
import importlib
from kivy.logger import Logger, LOG_LEVELS
from kivy.utils import platform
from kivy._version import __version__, RELEASE as _KIVY_RELEASE, \
from kivy.logger import file_log_handler
def kivy_usage():
    """Kivy Usage: %s [KIVY OPTION...] [-- PROGRAM OPTIONS]::

            Options placed after a '-- ' separator, will not be touched by kivy,
            and instead passed to your program.

            Set KIVY_NO_ARGS=1 in your environment or before you import Kivy to
            disable Kivy's argument parser.

        -h, --help
            Prints this help message.
        -d, --debug
            Shows debug log.
        -a, --auto-fullscreen
            Force 'auto' fullscreen mode (no resolution change).
            Uses your display's resolution. This is most likely what you want.
        -c, --config section:key[:value]
            Set a custom [section] key=value in the configuration object.
        -f, --fullscreen
            Force running in fullscreen mode.
        -k, --fake-fullscreen
            Force 'fake' fullscreen mode (no window border/decoration).
            Uses the resolution specified by width and height in your config.
        -w, --windowed
            Force running in a window.
        -p, --provider id:provider[,options]
            Add an input provider (eg: ccvtable1:tuio,192.168.0.1:3333).
        -m mod, --module=mod
            Activate a module (use "list" to get a list of available modules).
        -r, --rotation
            Rotate the window's contents (0, 90, 180, 270).
        -s, --save
            Save current Kivy configuration.
        --size=640x480
            Size of window geometry.
        --dpi=96
            Manually overload the Window DPI (for testing only.)
    """
    print(kivy_usage.__doc__ % basename(sys.argv[0]))