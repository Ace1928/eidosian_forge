import argparse
from keystoneauth1.identity.v3 import k2k
from keystoneauth1.loading import base
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from osc_lib import utils
def check_valid_authentication_options(options, auth_plugin_name):
    """Validate authentication options, and provide helpful error messages

    :param required_scope: indicate whether a scoped token is required

    """
    plugin_opts = base.get_plugin_options(auth_plugin_name)
    plugin_opts = {opt.dest: opt for opt in plugin_opts}
    msgs = []
    if not options.auth and auth_plugin_name != 'none':
        msgs.append(_('Set a cloud-name with --os-cloud or OS_CLOUD'))
    else:
        if 'password' in plugin_opts and (not (options.auth.get('username') or options.auth.get('user_id'))):
            msgs.append(_('Set a username with --os-username, OS_USERNAME, or auth.username or set a user-id with --os-user-id, OS_USER_ID, or auth.user_id'))
        if 'auth_url' in plugin_opts and (not options.auth.get('auth_url')):
            msgs.append(_('Set an authentication URL, with --os-auth-url, OS_AUTH_URL or auth.auth_url'))
        if 'url' in plugin_opts and (not options.auth.get('url')):
            msgs.append(_('Set a service URL, with --os-url, OS_URL or auth.url'))
        if 'token' in plugin_opts and (not options.auth.get('token')):
            msgs.append(_('Set a token with --os-token, OS_TOKEN or auth.token'))
    if msgs:
        raise exc.CommandError(_('Missing parameter(s): \n%s') % '\n'.join(msgs))