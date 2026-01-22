import sys
import weakref
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions
def testAromNAliphCRxn(self):
    m = Chem.MolFromSmiles('c1cccn1CCCC')
    res = RecapDecompose(m)
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 2)
    ks = res.GetLeaves().keys()
    self.assertTrue('*n1cccc1' in ks)
    self.assertTrue('*CCCC' in ks)
    m = Chem.MolFromSmiles('c1ccc2n1CCCC2')
    res = RecapDecompose(m)
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 0)