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
@utils.arg('id', metavar='<TASK_ID>', help=_('ID of task to describe.'))
def do_task_show(gc, args):
    """Describe a specific task."""
    task = gc.tasks.get(args.id)
    ignore = ['self', 'schema']
    task = dict([item for item in task.items() if item[0] not in ignore])
    utils.print_dict(task)