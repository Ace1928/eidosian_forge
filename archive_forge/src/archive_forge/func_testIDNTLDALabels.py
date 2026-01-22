import unittest
import idna
def testIDNTLDALabels(self):
    for ulabel, alabel in self.tld_strings:
        self.assertEqual(alabel, idna.alabel(ulabel))