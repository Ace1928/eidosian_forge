import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def driver_list(self, fields=None, params=''):
    """List baremetal drivers.

        :param List fields: List of fields to show
        :param String params: Additional kwargs
        :return: list of JSON driver objects
        """
    opts = self.get_opts(fields=fields)
    output = self.openstack('baremetal driver list {0} {1}'.format(opts, params))
    return json.loads(output)