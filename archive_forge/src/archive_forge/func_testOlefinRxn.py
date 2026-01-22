import sys
import weakref
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions
def testOlefinRxn(self):
    m = Chem.MolFromSmiles('ClC=CBr')
    res = RecapDecompose(m)
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 2)
    ks = res.GetLeaves().keys()
    self.assertTrue('*CCl' in ks)
    self.assertTrue('*CBr' in ks)
    m = Chem.MolFromSmiles('C1CC=CC1')
    res = RecapDecompose(m)
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 0)