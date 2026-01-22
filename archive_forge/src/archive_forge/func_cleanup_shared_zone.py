import sys
import tempfile
import traceback
import fixtures
from tempest.lib.exceptions import CommandFailed
from testtools.runtest import MultipleExceptions
from designateclient.functionaltests.client import DesignateCLI
@classmethod
def cleanup_shared_zone(cls, client, zone_id, shared_zone_id):
    try:
        client.unshare_zone(zone_id, shared_zone_id)
    except CommandFailed:
        pass