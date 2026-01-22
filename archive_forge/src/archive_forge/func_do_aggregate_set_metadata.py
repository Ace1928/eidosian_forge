import argparse
import collections
import datetime
import getpass
import logging
import os
import pprint
import sys
import time
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import timeutils
import novaclient
from novaclient import api_versions
from novaclient import base
from novaclient import client
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import shell
from novaclient import utils
from novaclient.v2 import availability_zones
from novaclient.v2 import quotas
from novaclient.v2 import servers
@utils.arg('aggregate', metavar='<aggregate>', help=_('Name or ID of aggregate to update.'))
@utils.arg('metadata', metavar='<key=value>', nargs='+', action='append', default=[], help=_('Metadata to add/update to aggregate. Specify only the key to delete a metadata item.'))
def do_aggregate_set_metadata(cs, args):
    """Update the metadata associated with the aggregate."""
    aggregate = _find_aggregate(cs, args.aggregate)
    metadata = _extract_metadata(args)
    currentmetadata = getattr(aggregate, 'metadata', {})
    if set(metadata.items()) & set(currentmetadata.items()):
        raise exceptions.CommandError(_('metadata already exists'))
    for key, value in metadata.items():
        if value is None and key not in currentmetadata:
            raise exceptions.CommandError(_('metadata key %s does not exist hence can not be deleted') % key)
    aggregate = cs.aggregates.set_metadata(aggregate.id, metadata)
    print(_('Metadata has been successfully updated for aggregate %s.') % aggregate.id)
    _print_aggregate_details(cs, aggregate)