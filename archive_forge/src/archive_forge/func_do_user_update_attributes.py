import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', help=_('ID or name of the instance.'))
@utils.arg('name', metavar='<name>', help=_('Name of user.'))
@utils.arg('--host', metavar='<host>', default=None, help=_('Optional host of user.'))
@utils.arg('--new_name', metavar='<new_name>', default=None, help=_('Optional new name of user.'))
@utils.arg('--new_password', metavar='<new_password>', default=None, help=_('Optional new password of user.'))
@utils.arg('--new_host', metavar='<new_host>', default=None, help=_('Optional new host of user.'))
@utils.service_type('database')
def do_user_update_attributes(cs, args):
    """Updates a user's attributes on an instance.
    At least one optional argument must be provided.
    """
    instance, _ = _find_instance_or_cluster(cs, args.instance)
    new_attrs = {}
    if args.new_name:
        new_attrs['name'] = args.new_name
    if args.new_password:
        new_attrs['password'] = args.new_password
    if args.new_host:
        new_attrs['host'] = args.new_host
    cs.users.update_attributes(instance, args.name, newuserattr=new_attrs, hostname=args.host)