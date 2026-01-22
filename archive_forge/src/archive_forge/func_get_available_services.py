import copy
import os
import botocore.session
from botocore.client import Config
from botocore.exceptions import DataNotFoundError, UnknownServiceError
import boto3
import boto3.utils
from boto3.exceptions import ResourceNotExistsError, UnknownAPIVersionError
from .resources.factory import ResourceFactory
def get_available_services(self):
    """
        Get a list of available services that can be loaded as low-level
        clients via :py:meth:`Session.client`.

        :rtype: list
        :return: List of service names
        """
    return self._session.get_available_services()