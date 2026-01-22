from unittest import TestCase
from ipykernel.tests import utils
from nbformat.converter import convert
from nbformat.reader import reads
import re
import json
from copy import copy
import unittest
def dump_canonical(self, obj):
    return json.dumps(obj, indent=2, sort_keys=True)