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
@utils.arg('namespace', metavar='<NAMESPACE>', help=_('Name of the namespace to which the tag belongs.'))
@utils.arg('tag', metavar='<TAG>', help=_('Name of the old tag.'))
@utils.arg('--name', metavar='<NAME>', default=None, required=True, help=_('New name of the new tag.'))
def do_md_tag_update(gc, args):
    """Rename a metadata definitions tag inside a namespace."""
    name = args.name.strip()
    if name:
        fields = {'name': name}
        new_tag = gc.metadefs_tag.update(args.namespace, args.tag, **fields)
        _tag_show(new_tag)
    else:
        utils.exit('Please supply at least one non-blank tag name.')