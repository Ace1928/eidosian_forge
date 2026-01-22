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
@utils.arg('namespace', metavar='<NAMESPACE>', help=_('Name of namespace.'))
def do_md_namespace_resource_type_list(gc, args):
    """List resource types associated to specific namespace."""
    resource_types = gc.metadefs_resource_type.get(args.namespace)
    utils.print_list(resource_types, ['name', 'prefix', 'properties_target'])