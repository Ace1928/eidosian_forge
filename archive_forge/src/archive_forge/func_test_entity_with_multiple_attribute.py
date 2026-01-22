import unittest
from prov.model import ProvDocument
from prov.tests.utility import RoundTripTestCase
from prov.tests.test_model import (
import os
from glob import glob
import logging
from prov.tests import examples
import prov.model as pm
import rdflib as rl
from rdflib.compare import graph_diff
from io import BytesIO, StringIO
@unittest.expectedFailure
def test_entity_with_multiple_attribute(self):
    TestAttributesBase.test_entity_with_multiple_attribute(self)