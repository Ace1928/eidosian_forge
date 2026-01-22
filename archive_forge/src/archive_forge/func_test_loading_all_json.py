import unittest
import logging
import os
from prov.model import ProvDocument, ProvBundle, ProvException, first, Literal
from prov.tests import examples
from prov.tests.attributes import TestAttributesBase
from prov.tests.qnames import TestQualifiedNamesBase
from prov.tests.statements import TestStatementsBase
from prov.tests.utility import RoundTripTestCase
def test_loading_all_json(self):
    for filename in self.fails:
        filepath = self.json_path + filename
        with open(filepath) as json_file:
            logger.info('Loading %s...', filepath)
            g1 = ProvDocument.deserialize(json_file)
            json_str = g1.serialize(indent=4)
            g2 = ProvDocument.deserialize(content=json_str)
            self.assertEqual(g1, g2, 'Round-trip JSON encoding/decoding failed:  %s.' % filename)