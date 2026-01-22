import unittest
import logging
import os
from prov.model import ProvDocument, ProvBundle, ProvException, first, Literal
from prov.tests import examples
from prov.tests.attributes import TestAttributesBase
from prov.tests.qnames import TestQualifiedNamesBase
from prov.tests.statements import TestStatementsBase
from prov.tests.utility import RoundTripTestCase
def document_1(self):
    d1 = ProvDocument()
    ns_ex = d1.add_namespace('ex', EX_URI)
    d1.entity(ns_ex['e1'])
    return d1