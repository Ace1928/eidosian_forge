import yaml
from oslo_serialization import jsonutils
from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
from zunclient import exceptions as exc
@utils.arg('registry', metavar='<registry>', help='ID or name of the registry to update.')
@utils.arg('--username', metavar='<username>', help='The new username for the registry.')
@utils.arg('--password', metavar='<password>', help='The new password for the registry.')
@utils.arg('--domain', metavar='<domain>', help='The new domain for the registry.')
@utils.arg('--name', metavar='<name>', help='The new name for the registry')
def do_registry_update(cs, args):
    """Update one or more attributes of the registry."""
    opts = {}
    opts['username'] = args.username
    opts['password'] = args.password
    opts['domain'] = args.domain
    opts['name'] = args.name
    opts = zun_utils.remove_null_parms(**opts)
    if not opts:
        raise exc.CommandError('You must update at least one property')
    registry = cs.registries.update(args.registry, **opts)
    _show_registry(registry)