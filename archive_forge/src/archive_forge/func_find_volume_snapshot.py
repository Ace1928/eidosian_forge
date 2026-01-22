import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def find_volume_snapshot(cs, snapshot):
    """Gets a volume snapshot by name or ID."""
    return utils.find_resource(cs.volume_snapshots, snapshot)