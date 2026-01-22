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
@utils.arg('--url', metavar='<URL>', required=True, help=_('URL of location to update.'))
@utils.arg('--metadata', metavar='<STRING>', default='{}', help=_('Metadata associated with the location. Must be a valid JSON object (default: %(default)s)'))
@utils.arg('id', metavar='<IMAGE_ID>', help=_('ID of image whose location is to be updated.'))
def do_location_update(gc, args):
    """Update metadata of an image's location."""
    try:
        metadata = json.loads(args.metadata)
        if metadata == {}:
            print("WARNING -- The location's metadata will be updated to an empty JSON object.")
    except ValueError:
        utils.exit('Metadata is not a valid JSON object.')
    else:
        image = gc.images.update_location(args.id, args.url, metadata)
        utils.print_dict(image)