import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.service_type('database')
def do_datastore_list(cs, args):
    """Lists available datastores."""
    datastores = cs.datastores.list()
    utils.print_list(datastores, ['id', 'name'])