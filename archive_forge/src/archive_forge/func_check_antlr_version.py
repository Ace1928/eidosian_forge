import os
import subprocess
import glob
from sympy.utilities.misc import debug
def check_antlr_version():
    debug('Checking antlr4 version...')
    try:
        debug(subprocess.check_output(['antlr4']).decode('utf-8').split('\n')[0])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        debug("The 'antlr4' command line tool is not installed, or not on your PATH.\n> Please refer to the README.md file for more information.")
        return False