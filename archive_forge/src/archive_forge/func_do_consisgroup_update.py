import argparse
import collections
import copy
import os
from oslo_utils import strutils
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3 import availability_zones
@utils.arg('consistencygroup', metavar='<consistencygroup>', help='Name or ID of a consistency group.')
@utils.arg('--name', metavar='<name>', help='New name for consistency group. Default=None.')
@utils.arg('--description', metavar='<description>', help='New description for consistency group. Default=None.')
@utils.arg('--add-volumes', metavar='<uuid1,uuid2,......>', help='UUID of one or more volumes to be added to the consistency group, separated by commas. Default=None.')
@utils.arg('--remove-volumes', metavar='<uuid3,uuid4,......>', help='UUID of one or more volumes to be removed from the consistency group, separated by commas. Default=None.')
def do_consisgroup_update(cs, args):
    """Updates a consistency group."""
    kwargs = {}
    if args.name is not None:
        kwargs['name'] = args.name
    if args.description is not None:
        kwargs['description'] = args.description
    if args.add_volumes is not None:
        kwargs['add_volumes'] = args.add_volumes
    if args.remove_volumes is not None:
        kwargs['remove_volumes'] = args.remove_volumes
    if not kwargs:
        msg = 'At least one of the following args must be supplied: name, description, add-volumes, remove-volumes.'
        raise exceptions.ClientException(code=1, message=msg)
    shell_utils.find_consistencygroup(cs, args.consistencygroup).update(**kwargs)
    print("Request to update consistency group '%s' has been accepted." % args.consistencygroup)