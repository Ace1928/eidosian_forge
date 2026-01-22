import errno
import os
import pdb
import socket
import stat
import struct
import sys
import time
import traceback
import gflags as flags
def parse_flags_with_usage(args):
    """Try parsing the flags, printing usage and exiting if unparseable."""
    try:
        argv = FLAGS(args)
        return argv
    except flags.FlagsError as error:
        sys.stderr.write('FATAL Flags parsing error: %s\n' % error)
        sys.stderr.write('Pass --help or --helpshort to see help on flags.\n')
        sys.exit(1)