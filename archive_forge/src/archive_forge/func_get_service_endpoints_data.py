import copy
import logging
import re
from enum import Enum
from botocore import UNSIGNED, xform_name
from botocore.auth import AUTH_TYPE_MAPS, HAS_CRT
from botocore.crt import CRT_SUPPORTED_AUTH_TYPES
from botocore.endpoint_provider import EndpointProvider
from botocore.exceptions import (
from botocore.utils import ensure_boolean, instance_cache
def get_service_endpoints_data(self, service_name, partition_name='aws'):
    for partition in self._endpoint_data['partitions']:
        if partition['partition'] != partition_name:
            continue
        services = partition['services']
        if service_name not in services:
            continue
        return services[service_name]['endpoints']