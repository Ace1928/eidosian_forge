from unittest import TestCase
from ipykernel.tests import utils
from nbformat.converter import convert
from nbformat.reader import reads
import re
import json
from copy import copy
import unittest
def should_continue(self, msg):
    """
        determine whether the current message is the last for this cell
        """
    if msg is None:
        return True
    return not (msg['msg_type'] == 'status' and msg['content']['execution_state'] == 'idle')