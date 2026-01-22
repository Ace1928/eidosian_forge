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
@utils.arg('image_id', metavar='<IMAGE_ID>', help=_('Image from which to display member.'))
@utils.arg('member_id', metavar='<MEMBER_ID>', help=_('Project to display.'))
def do_member_get(gc, args):
    """Show details of an image member"""
    member = gc.image_members.get(args.image_id, args.member_id)
    utils.print_dict(member)