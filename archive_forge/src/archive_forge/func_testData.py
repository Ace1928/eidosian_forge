from typing import Dict, List, Tuple
from twisted.internet.testing import StringTransport
from twisted.protocols import postfix
from twisted.trial import unittest
def testData(self):
    for entry in self.data:
        raw = entry[0]
        quoted = entry[1:]
        self.assertEqual(postfix.quote(raw), quoted[0])
        for q in quoted:
            self.assertEqual(postfix.unquote(q), raw)