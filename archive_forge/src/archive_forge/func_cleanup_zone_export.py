import sys
import tempfile
import traceback
import fixtures
from tempest.lib.exceptions import CommandFailed
from testtools.runtest import MultipleExceptions
from designateclient.functionaltests.client import DesignateCLI
@classmethod
def cleanup_zone_export(cls, client, zone_export_id):
    try:
        client.zone_export_delete(zone_export_id)
    except CommandFailed:
        pass