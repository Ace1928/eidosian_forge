import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def find_message(cs, message):
    """Gets a message by ID."""
    return utils.find_resource(cs.messages, message)