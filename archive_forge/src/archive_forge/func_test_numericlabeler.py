import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_numericlabeler(self):
    m = self.m
    lbl = NumericLabeler('x')
    self.assertEqual(lbl(m.mycomp), 'x1')
    self.assertEqual(lbl(m.mycomp), 'x2')
    self.assertEqual(lbl(m.that), 'x3')
    self.assertEqual(lbl(self.long1), 'x4')
    self.assertEqual(lbl(m.myblock), 'x5')
    self.assertEqual(lbl(m.myblock.mystreet), 'x6')
    self.assertEqual(lbl(self.thecopy), 'x7')
    lbl = NumericLabeler('xyz')
    self.assertEqual(lbl(m.mycomp), 'xyz1')
    self.assertEqual(lbl(m.mycomp), 'xyz2')
    self.assertEqual(lbl(m.that), 'xyz3')
    self.assertEqual(lbl(self.long1), 'xyz4')
    self.assertEqual(lbl(m.myblock), 'xyz5')
    self.assertEqual(lbl(m.myblock.mystreet), 'xyz6')
    self.assertEqual(lbl(self.thecopy), 'xyz7')