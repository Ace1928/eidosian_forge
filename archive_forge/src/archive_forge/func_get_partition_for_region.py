import copy
import logging
import os
import platform
import socket
import warnings
import botocore.client
import botocore.configloader
import botocore.credentials
import botocore.tokens
from botocore import (
from botocore.compat import HAS_CRT, MutableMapping
from botocore.configprovider import (
from botocore.errorfactory import ClientExceptionsFactory
from botocore.exceptions import (
from botocore.hooks import (
from botocore.loaders import create_loader
from botocore.model import ServiceModel
from botocore.parsers import ResponseParserFactory
from botocore.regions import EndpointResolver
from botocore.useragent import UserAgentString
from botocore.utils import (
from botocore.compat import HAS_CRT  # noqa
def get_partition_for_region(self, region_name):
    """Lists the partition name of a particular region.

        :type region_name: string
        :param region_name: Name of the region to list partition for (e.g.,
             us-east-1).

        :rtype: string
        :return: Returns the respective partition name (e.g., aws).
        """
    resolver = self._get_internal_component('endpoint_resolver')
    return resolver.get_partition_for_region(region_name)