from keystoneclient import exceptions as identity_exc
from keystoneclient.v3 import domains
from keystoneclient.v3 import groups
from keystoneclient.v3 import projects
from keystoneclient.v3 import users
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
def get_immutable_options(parsed_args):
    options = {}
    if parsed_args.immutable:
        options['immutable'] = True
    if parsed_args.no_immutable:
        options['immutable'] = False
    return options