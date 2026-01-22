import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def chassis_show(self, uuid, fields=None, params=''):
    """Show specified baremetal chassis.

        :param String uuid: UUID of the chassis
        :param List fields: List of fields to show
        :param List params: Additional kwargs
        :return: JSON object of chassis
        """
    opts = self.get_opts(fields)
    chassis = self.openstack('baremetal chassis show {0} {1} {2}'.format(opts, uuid, params))
    return json.loads(chassis)