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
def help_datamanagers(options):
    import pyomo.environ
    from pyomo.dataportal import DataManagerFactory
    wrapper = textwrap.TextWrapper()
    wrapper.initial_indent = '      '
    wrapper.subsequent_indent = '      '
    print('')
    print('Pyomo Data Managers')
    print('-------------------')
    for xform in sorted(DataManagerFactory):
        print('  ' + xform)
        print(wrapper.fill(DataManagerFactory.doc(xform)))