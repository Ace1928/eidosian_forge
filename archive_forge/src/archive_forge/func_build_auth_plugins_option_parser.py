import argparse
from keystoneauth1.identity.v3 import k2k
from keystoneauth1.loading import base
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from osc_lib import utils
def build_auth_plugins_option_parser(parser):
    """Auth plugins options builder

    Builds dynamically the list of options expected by each available
    authentication plugin.

    """
    available_plugins = list(get_plugin_list())
    parser.add_argument('--os-auth-type', metavar='<auth-type>', dest='auth_type', default=utils.env('OS_AUTH_TYPE'), help=_('Select an authentication type. Available types: %s. Default: selected based on --os-username/--os-token (Env: OS_AUTH_TYPE)') % ', '.join(available_plugins), choices=available_plugins)
    envs = {'OS_PROJECT_NAME': utils.env('OS_PROJECT_NAME', default=utils.env('OS_TENANT_NAME')), 'OS_PROJECT_ID': utils.env('OS_PROJECT_ID', default=utils.env('OS_TENANT_ID'))}
    for o in get_options_list():
        if 'tenant' not in o:
            parser.add_argument('--os-' + o, metavar='<auth-%s>' % o, dest=o.replace('-', '_'), default=envs.get(OPTIONS_LIST[o]['env'], utils.env(OPTIONS_LIST[o]['env'])), help=_('%(help)s\n(Env: %(env)s)') % {'help': OPTIONS_LIST[o]['help'], 'env': OPTIONS_LIST[o]['env']})
    parser.add_argument('--os-tenant-name', metavar='<auth-tenant-name>', dest='os_project_name', help=argparse.SUPPRESS)
    parser.add_argument('--os-tenant-id', metavar='<auth-tenant-id>', dest='os_project_id', help=argparse.SUPPRESS)
    return parser