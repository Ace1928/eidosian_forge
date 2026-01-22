import itertools
import re
from oslo_log import log as logging
from heat.api.aws import exception
def reformat_dict_keys(keymap=None, inputdict=None):
    """Utility function for mapping one dict format to another."""
    keymap = keymap or {}
    inputdict = inputdict or {}
    return dict([(outk, inputdict[ink]) for ink, outk in keymap.items() if ink in inputdict])