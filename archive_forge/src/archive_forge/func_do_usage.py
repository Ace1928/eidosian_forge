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
def do_usage(gc, args):
    """Get quota usage information."""
    columns = ['Quota', 'Limit', 'Usage']
    usage = gc.info.get_usage()
    utils.print_dict_list([dict(v, quota=k) for k, v in usage.items()], columns)