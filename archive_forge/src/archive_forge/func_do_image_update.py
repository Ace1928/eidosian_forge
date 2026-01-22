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
@utils.arg('id', metavar='<IMAGE_ID>', help=_('ID of image to update.'))
@utils.schema_args(get_image_schema, omit=['id', 'locations', 'tags', 'os_hidden'])
@utils.arg('--hidden', type=strutils.bool_from_string, metavar='[True|False]', default=None, dest='os_hidden', help='If true, image will not appear in default image list response.')
@utils.arg('--property', metavar='<key=value>', action='append', default=[], help=_('Arbitrary property to associate with image. May be used multiple times.'))
@utils.arg('--remove-property', metavar='key', action='append', default=[], help=_('Name of arbitrary property to remove from the image.'))
def do_image_update(gc, args):
    """Update an existing image."""
    schema = gc.schemas.get('image')
    _args = [(x[0].replace('-', '_'), x[1]) for x in vars(args).items()]
    fields = dict(filter(lambda x: x[1] is not None and (x[0] in ['property', 'remove_property'] or schema.is_core_property(x[0])), _args))
    raw_properties = fields.pop('property', [])
    for datum in raw_properties:
        key, value = datum.split('=', 1)
        fields[key] = value
    remove_properties = fields.pop('remove_property', None)
    image_id = fields.pop('id')
    image = gc.images.update(image_id, remove_properties, **fields)
    utils.print_image(image)