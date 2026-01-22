import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('module', metavar='<module>', type=str, help=_('Name or ID of the module.'))
@utils.arg('--name', metavar='<name>', type=str, default=None, help=_('Name of the module.'))
@utils.arg('--type', metavar='<type>', type=str, default=None, help=_('Type of the module. The type must be supported by a corresponding module driver plugin on the datastore it is applied to.'))
@utils.arg('--file', metavar='<filename>', type=argparse.FileType('rb', 0), default=None, help=_('File containing data contents for the module.'))
@utils.arg('--description', metavar='<description>', type=str, default=None, help=_('Description of the module.'))
@utils.arg('--datastore', metavar='<datastore>', default=None, help=_('Name or ID of datastore this module can be applied to. If not specified, module can be applied to all datastores.'))
@utils.arg('--all_datastores', default=None, action='store_const', const=True, help=_('Module is valid for all datastores.'))
@utils.arg('--datastore_version', metavar='<version>', default=None, help=_('Name or ID of datastore version this module can be applied to. If not specified, module can be applied to all versions.'))
@utils.arg('--all_datastore_versions', default=None, action='store_const', const=True, help=_('Module is valid for all datastore versions.'))
@utils.arg('--auto_apply', action='store_true', default=None, help=_('Automatically apply this module when creating an instance or cluster. Admin only.'))
@utils.arg('--no_auto_apply', dest='auto_apply', action='store_false', default=None, help=_('Do not automatically apply this module when creating an instance or cluster. Admin only.'))
@utils.arg('--all_tenants', action='store_true', default=None, help=_('Module is valid for all tenants. Admin only.'))
@utils.arg('--no_all_tenants', dest='all_tenants', action='store_false', default=None, help=_('Module is valid for current tenant only. Admin only.'))
@utils.arg('--hidden', action='store_true', default=None, help=_('Hide this module from non-admin users. Useful in creating auto-apply modules without cluttering up module lists. Admin only.'))
@utils.arg('--no_hidden', dest='hidden', action='store_false', default=None, help=_('Allow all users to see this module. Admin only.'))
@utils.arg('--live_update', action='store_true', default=None, help=_('Allow module to be updated or deleted even if it is already applied to a current instance or cluster.'))
@utils.arg('--no_live_update', dest='live_update', action='store_false', default=None, help=_('Restricts a module from being updated or deleted if it is already applied to a current instance or cluster.'))
@utils.arg('--priority_apply', action='store_true', default=None, help=_('Sets a priority for applying the module. All priority modules will be applied before non-priority ones. Admin only.'))
@utils.arg('--no_priority_apply', dest='priority_apply', action='store_false', default=None, help=_('Removes apply priority from the module. Admin only.'))
@utils.arg('--apply_order', type=int, default=None, choices=range(0, 10), help=_('Sets an order for applying the module. Modules with a lower value will be applied before modules with a higher value. Modules having the same value may be applied in any order (default %(default)s).'))
@utils.arg('--full_access', action='store_true', default=None, help=_("Marks a module as 'non-admin', unless an admin-only option was specified. Admin only."))
@utils.arg('--no_full_access', dest='full_access', action='store_false', default=None, help=_('Restricts modification access for non-admin. Admin only.'))
@utils.service_type('database')
def do_module_update(cs, args):
    """Update a module."""
    module = _find_module(cs, args.module)
    contents = args.file.read() if args.file else None
    visible = not args.hidden if args.hidden is not None else None
    datastore_args = {'datastore': args.datastore, 'datastore_version': args.datastore_version}
    updated_module = cs.modules.update(module, name=args.name, module_type=args.type, contents=contents, description=args.description, all_tenants=args.all_tenants, auto_apply=args.auto_apply, visible=visible, live_update=args.live_update, all_datastores=args.all_datastores, all_datastore_versions=args.all_datastore_versions, priority_apply=args.priority_apply, apply_order=args.apply_order, full_access=args.full_access, **datastore_args)
    _print_object(updated_module)