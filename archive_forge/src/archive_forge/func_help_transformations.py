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
def help_transformations():
    import pyomo.environ
    from pyomo.core import TransformationFactory
    wrapper = textwrap.TextWrapper()
    wrapper.initial_indent = '      '
    wrapper.subsequent_indent = '      '
    print('')
    print('Pyomo Model Transformations')
    print('---------------------------')
    for xform in sorted(TransformationFactory):
        print('  ' + xform)
        _doc = TransformationFactory.doc(xform) or ''
        _init_doc = TransformationFactory.get_class(xform).__init__.__doc__ or ''
        if _init_doc.strip().startswith('DEPRECATED') and 'DEPRECAT' not in _doc:
            _doc = ' '.join(('[DEPRECATED]', _doc))
        if _doc:
            print(wrapper.fill(_doc))