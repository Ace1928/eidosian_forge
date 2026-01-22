from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient.i18n import _
@utils.arg('--project-id', required=True, metavar='<project-id>', help=_('Project Id.'))
@utils.arg('--resource', required=True, metavar='<resource>', help=_('Resource name.'))
@utils.arg('--hard-limit', metavar='<hard-limit>', type=int, default=1, help=_('Max resource limit.'))
@utils.deprecated(utils.MAGNUM_CLIENT_DEPRECATION_WARNING)
def do_quotas_create(cs, args):
    """Create a quota."""
    opts = dict()
    opts['project_id'] = args.project_id
    opts['resource'] = args.resource
    opts['hard_limit'] = args.hard_limit
    try:
        quota = cs.quotas.create(**opts)
        _show_quota(quota)
    except Exception as e:
        print('Create quota for project_id %(id)s resource %(res)s failed: %(e)s' % {'id': args.project_id, 'res': args.resource, 'e': e.details})