import json
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional import base
def port_delete(self, uuid, ignore_exceptions=False):
    """Try to delete baremetal port by UUID.

        :param String uuid: UUID of the port
        :param Bool ignore_exceptions: Ignore exception (needed for cleanUp)
        :return: raw values output
        :raise: CommandFailed exception when command fails to delete a port
        """
    try:
        return self.openstack('baremetal port delete {0}'.format(uuid))
    except exceptions.CommandFailed:
        if not ignore_exceptions:
            raise