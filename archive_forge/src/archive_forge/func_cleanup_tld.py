import sys
import tempfile
import traceback
import fixtures
from tempest.lib.exceptions import CommandFailed
from testtools.runtest import MultipleExceptions
from designateclient.functionaltests.client import DesignateCLI
@classmethod
def cleanup_tld(cls, client, tld_id):
    try:
        client.tld_delete(tld_id)
    except CommandFailed:
        pass