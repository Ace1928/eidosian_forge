import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_case_sensitive_shortnamelabeler(self):
    m = self.m
    lbl = ShortNameLabeler(20, '_')
    self.assertEqual(lbl(m.mycomp), 'mycomp')
    self.assertEqual(lbl(m.that), 'that')
    self.assertEqual(lbl(self.long1), 'longcomponentname_1_')
    self.assertEqual(lbl(self.long2), 'nentnamerighthere_2_')
    self.assertEqual(lbl(self.long3), 'ngonebutdifferent_3_')
    self.assertEqual(lbl(self.long4), 'longcomponentname_4_')
    self.assertEqual(lbl(self.long5), 'gcomponentname_1__5_')
    self.assertEqual(lbl(m.myblock), 'myblock')
    self.assertEqual(lbl(m.myblock.mystreet), 'myblock_mystreet')
    self.assertEqual(lbl(m.ind[3]), 'ind_3_')
    self.assertEqual(lbl(m.ind[10]), 'ind_10_')
    self.assertEqual(lbl(m.ind[1]), 'ind_1_')
    self.assertEqual(lbl(self.thecopy), '_myblock_mystreet_')
    m._myblock = Block()
    m._myblock.mystreet_ = Var()
    self.assertEqual(lbl(m.mycomp), 'mycomp_6_')
    self.assertEqual(lbl(m._myblock.mystreet_), 'myblock_mystreet__7_')
    self.assertEqual(lbl(m.MyComp), 'MyComp')