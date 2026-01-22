import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.service_type('database')
def do_limit_list(cs, args):
    """Lists the limits for a tenant."""
    limits = cs.limits.list()
    absolute = limits.pop(0)
    _print_object(absolute)
    utils.print_list(limits, ['value', 'verb', 'remaining', 'unit'])