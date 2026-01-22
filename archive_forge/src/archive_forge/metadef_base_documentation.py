from oslo_utils.fixture import uuidsentinel as uuids
import requests
from glance.tests import functional
Create a metadef namespace.

        :param path: string representing the namespace API path
        :param headers: dictionary with the headers to use for the request
        :param namespace: dictionary representing the namespace to create

        :returns: a dictionary of the namespace in the response
        