from oslo_serialization import jsonutils
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from zunclient.tests.functional import base
def container_list(self, fields=None, params=''):
    """List Container.

        :param List fields: List of fields to show
        :param String params: Additional kwargs
        :return: list of JSON container objects
        """
    opts = self.get_opts(fields=fields)
    output = self.openstack('appcontainer list {0} {1}'.format(opts, params))
    return jsonutils.loads(output)