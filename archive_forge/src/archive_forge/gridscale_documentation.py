import time
from libcloud.compute.base import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.gridscale import GridscaleBaseDriver, GridscaleConnection
from libcloud.compute.providers import Provider

        Get specific uuid specific resource.

        :param endpoint_suffix: Endpoint resource e.g. servers/.
        :type endpoint_suffix: ``str``
        :param object_uuid: Uuid of resource to be pulled.
        :type object_uuid: ``str``
        :return: Response dictionary.
        :rtype: nested ``dict``
        