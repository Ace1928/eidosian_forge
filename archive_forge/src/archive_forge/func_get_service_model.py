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
def get_service_model(self, service_name, api_version=None):
    """Get the service model object.

        :type service_name: string
        :param service_name: The service name

        :type api_version: string
        :param api_version: The API version of the service.  If none is
            provided, then the latest API version will be used.

        :rtype: L{botocore.model.ServiceModel}
        :return: The botocore service model for the service.

        """
    service_description = self.get_service_data(service_name, api_version)
    return ServiceModel(service_description, service_name=service_name)