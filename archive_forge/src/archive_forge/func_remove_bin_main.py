import argparse
import os
from os import path as op
import shutil
import sys
from . import plugins
from .core import util
def remove_bin_main():
    """Argument-parsing wrapper for `remove_bin`"""
    description = 'Remove plugin binary dependencies'
    phelp = 'Plugin name for which to remove the binary. ' + 'If no argument is given, all binaries are removed.'
    example_text = 'examples:\n' + '  imageio_remove_bin all\n' + '  imageio_remove_bin freeimage\n'
    parser = argparse.ArgumentParser(description=description, epilog=example_text, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('plugin', type=str, nargs='*', default='all', help=phelp)
    args = parser.parse_args()
    remove_bin(plugin_names=args.plugin)