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
def do_cache_list(gc, args):
    """Get cache state."""
    if not gc.endpoint_provided:
        utils.exit('Direct server endpoint needs to be provided. Do not use loadbalanced or catalog endpoints.')
    cached_images = gc.cache.list()
    utils.print_cached_images(cached_images)