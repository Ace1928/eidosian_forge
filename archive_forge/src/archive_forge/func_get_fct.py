import difflib
import glob
import inspect
import io
from lxml import etree
import os
import unittest
import warnings
from prov.identifier import Namespace, QualifiedName
from prov.constants import PROV
import prov.model as prov
from prov.tests.test_model import AllTestsBase
from prov.tests.utility import RoundTripTestCase
def get_fct(f):
    if name in ['pc1']:
        force_types = True
    else:
        force_types = False

    def fct(self):
        self._perform_round_trip(f, force_types=force_types)
    return fct