import os
from troveclient import base
from troveclient import common
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules as core_modules
from swiftclient import client as swift_client
def promote_to_replica_source(self, instance):
    """Promote a replica to be the new replica_source of its set

        :param instance: The :class:`Instance` (or its ID) of the database
                         instance to promote.
        """
    body = {'promote_to_replica_source': {}}
    self._action(instance, body)