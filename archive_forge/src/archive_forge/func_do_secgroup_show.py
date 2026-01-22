import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('security_group', metavar='<security_group>', help=_('Security group ID.'))
@utils.service_type('database')
def do_secgroup_show(cs, args):
    """Shows details of a security group."""
    sec_grp = cs.security_groups.get(args.security_group)
    del sec_grp._info['rules']
    _print_object(sec_grp)