import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def node_show(self, identifier, fields=None, params=''):
    """Show specified baremetal node.

        :param String identifier: Name or UUID of the node
        :param List fields: List of fields to show
        :param List params: Additional kwargs
        :return: JSON object of node
        """
    opts = self.get_opts(fields)
    output = self.openstack('baremetal node show {0} {1} {2}'.format(opts, identifier, params))
    return json.loads(output)