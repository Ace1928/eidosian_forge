import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def print_volume_snapshot(snapshot):
    print_dict(snapshot._info)