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
def help_writers():
    import pyomo.environ
    from pyomo.opt.base import WriterFactory
    wrapper = textwrap.TextWrapper()
    wrapper.initial_indent = '      '
    wrapper.subsequent_indent = '      '
    print('')
    print('Pyomo Problem Writers')
    print('---------------------')
    for writer in sorted(WriterFactory):
        print('  ' + writer)
        print(wrapper.fill(WriterFactory.doc(writer)))