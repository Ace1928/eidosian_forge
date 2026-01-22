import ast
from tempest.lib.cli import output_parser
import testtools
from manilaclient import api_versions
from manilaclient import config
def get_default_subnet(client, share_network_id):
    return get_subnet_by_availability_zone_name(client, share_network_id, 'None')