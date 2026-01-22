import os
from troveclient import base
from troveclient import common
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules as core_modules
from swiftclient import client as swift_client
def log_list(self, instance):
    """Get a list of all guest logs.

        :param instance: The :class:`Instance` (or its ID) of the database
                         instance to get the log for.
        :rtype: list of :class:`DatastoreLog`.
        """
    url = '/instances/%s/log' % base.getid(instance)
    resp, body = self.api.client.get(url)
    common.check_for_exceptions(resp, body, url)
    return [DatastoreLog(self, log, loaded=True) for log in body['logs']]