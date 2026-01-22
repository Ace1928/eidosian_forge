import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def translate_volume_snapshot_keys(collection):
    convert = [('volumeId', 'volume_id')]
    translate_keys(collection, convert)