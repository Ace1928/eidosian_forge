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
@utils.arg('namespace', metavar='<NAMESPACE>', help=_('Name of namespace the property belongs.'))
@utils.arg('property', metavar='<PROPERTY>', help=_('Name of a property.'))
@utils.arg('--name', metavar='<NAME>', default=None, help=_('New name of a property.'))
@utils.arg('--title', metavar='<TITLE>', default=None, help=_('Property name displayed to the user.'))
@utils.arg('--schema', metavar='<SCHEMA>', default=None, help=_('Valid JSON schema of a property.'))
def do_md_property_update(gc, args):
    """Update metadata definitions property inside a namespace."""
    fields = {}
    if args.name:
        fields['name'] = args.name
    if args.title:
        fields['title'] = args.title
    if args.schema:
        try:
            schema = json.loads(args.schema)
        except ValueError:
            utils.exit('Schema is not a valid JSON object.')
        else:
            fields.update(schema)
    new_property = gc.metadefs_property.update(args.namespace, args.property, **fields)
    utils.print_dict(new_property)