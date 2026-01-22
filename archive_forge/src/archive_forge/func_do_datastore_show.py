import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('datastore', metavar='<datastore>', help=_('ID of the datastore.'))
@utils.service_type('database')
def do_datastore_show(cs, args):
    """Shows details of a datastore."""
    datastore = cs.datastores.get(args.datastore)
    info = datastore._info.copy()
    versions = info.get('versions', [])
    versions_str = '\n'.join([ver['name'] + ' (' + ver['id'] + ')' for ver in versions])
    info['versions (id)'] = versions_str
    info.pop('versions', None)
    info.pop('links', None)
    if hasattr(datastore, 'default_version'):
        def_ver_id = getattr(datastore, 'default_version')
        info['default_version'] = [ver['name'] for ver in versions if ver['id'] == def_ver_id][0]
    utils.print_dict(info)