import os
import argparse
import inspect
import sys
from ..interfaces.base import Interface, InputMultiPath, traits
from ..interfaces.base.support import get_trait_desc
from .misc import str2bool
def listClasses(module=None):
    if module:
        __import__(module)
        pkg = sys.modules[module]
        print('Available Interfaces:')
        for k, v in sorted(list(pkg.__dict__.items())):
            if inspect.isclass(v) and issubclass(v, Interface):
                print('\t%s' % k)