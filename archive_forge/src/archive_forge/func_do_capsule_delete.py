import yaml
from oslo_serialization import jsonutils
from zunclient.common import cliutils as utils
from zunclient.common import template_utils
from zunclient.common import utils as zun_utils
from zunclient.i18n import _
@utils.arg('capsules', metavar='<capsule>', nargs='+', help='ID or name of the (capsule)s to delete.')
def do_capsule_delete(cs, args):
    """Delete specified capsules."""
    for capsule in args.capsules:
        try:
            cs.capsules.delete(capsule)
            print('Request to delete capsule %s has been accepted.' % capsule)
        except Exception as e:
            print('Delete for capsule %(capsule)s failed: %(e)s' % {'capsule': capsule, 'e': e})