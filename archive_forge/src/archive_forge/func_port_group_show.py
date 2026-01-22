import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def port_group_show(self, identifier, fields=None, params=''):
    """Show specified baremetal port group.

        :param String identifier: Name or UUID of the port group
        :param List fields: List of fields to show
        :param List params: Additional kwargs
        :return: JSON object of port group
        """
    opts = self.get_opts(fields)
    output = self.openstack('baremetal port group show {0} {1} {2}'.format(identifier, opts, params))
    return json.loads(output)