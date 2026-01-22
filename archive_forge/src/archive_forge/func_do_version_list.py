from zunclient import api_versions
from zunclient.common import cliutils as utils
def do_version_list(cs, args):
    """List all API versions."""
    print('Client supported API versions:')
    print('Minimum version %(v)s' % {'v': api_versions.MIN_API_VERSION})
    print('Maximum version %(v)s' % {'v': api_versions.MAX_API_VERSION})
    print('\nServer supported API versions:')
    result = cs.versions.list()
    columns = ['Id', 'Status', 'Min Version', 'Max Version']
    utils.print_list(result, columns)