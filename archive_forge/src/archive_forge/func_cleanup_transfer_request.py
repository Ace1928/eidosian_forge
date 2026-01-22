import sys
import tempfile
import traceback
import fixtures
from tempest.lib.exceptions import CommandFailed
from testtools.runtest import MultipleExceptions
from designateclient.functionaltests.client import DesignateCLI
@classmethod
def cleanup_transfer_request(cls, client, transfer_request_id):
    try:
        client.zone_transfer_request_delete(transfer_request_id)
    except CommandFailed:
        pass