import os
import os.path
import sys
import glob
import textwrap
import logging
import socket
import subprocess
import pyomo.common
from pyomo.common.collections import Bunch
import pyomo.scripting.pyomo_parser
def command_exec(options):
    cmddir = os.path.dirname(os.path.abspath(sys.executable)) + os.sep
    if options.summary:
        print('')
        print('The following commands are installed in the Pyomo bin directory:')
        print('----------------------------------------------------------------')
        for file in sorted(glob.glob(cmddir + '*')):
            print(' ' + os.path.basename(file))
        print('')
        if len(options.command) > 0:
            print('WARNING: ignoring command specification')
        return
    if len(options.command) == 0:
        print('  ERROR: no command specified')
        return 1
    if not os.path.exists(cmddir + options.command[0]):
        print("  ERROR: the command '%s' does not exist" % (cmddir + options.command[0]))
        return 1
    return subprocess.run([cmddir] + options.command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode