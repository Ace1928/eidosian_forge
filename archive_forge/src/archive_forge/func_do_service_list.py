from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
def do_service_list(cs, args):
    """Print a list of magnum services."""
    mservices = cs.mservices.list()
    columns = ('id', 'host', 'binary', 'state', 'disabled', 'disabled_reason', 'created_at', 'updated_at')
    utils.print_list(mservices, columns, {'versions': magnum_utils.print_list_field('versions')})