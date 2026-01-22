import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def find_cgsnapshot(cs, cgsnapshot):
    """Gets a cgsnapshot by name or ID."""
    return utils.find_resource(cs.cgsnapshots, cgsnapshot)