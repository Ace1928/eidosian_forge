import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def find_consistencygroup(cs, consistencygroup):
    """Gets a consistency group by name or ID."""
    return utils.find_resource(cs.consistencygroups, consistencygroup)