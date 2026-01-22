import json
import os
import sys
from oslo_utils import strutils
from glanceclient._i18n import _
from glanceclient.common import progressbar
from glanceclient.common import utils
from glanceclient import exc
from glanceclient.v2 import cache
from glanceclient.v2 import image_members
from glanceclient.v2 import image_schema
from glanceclient.v2 import images
from glanceclient.v2 import namespace_schema
from glanceclient.v2 import resource_type_schema
from glanceclient.v2 import tasks
@utils.arg('image_id', metavar='<IMAGE_ID>', help=_('Image from which to update member.'))
@utils.arg('member_id', metavar='<MEMBER_ID>', help=_('Tenant to update.'))
@utils.arg('member_status', metavar='<MEMBER_STATUS>', choices=MEMBER_STATUS_VALUES, help=_('Updated status of member. Valid Values: %s') % ', '.join((str(val) for val in MEMBER_STATUS_VALUES)))
def do_member_update(gc, args):
    """Update the status of a member for a given image."""
    if not (args.image_id and args.member_id and args.member_status):
        utils.exit('Unable to update member. Specify image_id, member_id and member_status')
    else:
        member = gc.image_members.update(args.image_id, args.member_id, args.member_status)
        member = [member]
        columns = ['Image ID', 'Member ID', 'Status']
        utils.print_list(member, columns)