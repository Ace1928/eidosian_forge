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
@utils.arg('image_id', metavar='<IMAGE_ID>', help=_('Image with which to create member.'))
@utils.arg('member_id', metavar='<MEMBER_ID>', help=_('Tenant to add as member.'))
def do_member_create(gc, args):
    """Create member for a given image."""
    if not (args.image_id and args.member_id):
        utils.exit('Unable to create member. Specify image_id and member_id')
    else:
        member = gc.image_members.create(args.image_id, args.member_id)
        member = [member]
        columns = ['Image ID', 'Member ID', 'Status']
        utils.print_list(member, columns)