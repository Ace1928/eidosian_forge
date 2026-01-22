import os
from troveclient import base
from troveclient import common
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules as core_modules
from swiftclient import client as swift_client
def detach_replica(self):
    """Stops the replica database from being replicated to."""
    self.manager.edit(self.id, detach_replica_source=True)