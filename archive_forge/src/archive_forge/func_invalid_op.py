import getpass
import argparse
import sys
from . import core
from . import backend
from . import completion
from . import set_keyring, get_password, set_password, delete_password
from .util import platform_
def invalid_op(self):
    self.parser.error(f'Specify operation ({', '.join(self.parser._operations)}).')