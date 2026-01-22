import sys
import weakref
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions
def testMinFragmentSize(self):
    m = Chem.MolFromSmiles('CCCOCCC')
    res = RecapDecompose(m)
    self.assertTrue(res)
    self.assertTrue(res.children == {})
    res = RecapDecompose(m, minFragmentSize=3)
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 1)
    ks = res.GetLeaves().keys()
    self.assertTrue('*CCC' in ks)
    m = Chem.MolFromSmiles('CCCOCC')
    res = RecapDecompose(m, minFragmentSize=3)
    self.assertTrue(res)
    self.assertTrue(res.children == {})
    m = Chem.MolFromSmiles('CCCOCCOC')
    res = RecapDecompose(m, minFragmentSize=2)
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 2)
    ks = res.GetLeaves().keys()
    self.assertTrue('*CCC' in ks)
    ks = res.GetLeaves().keys()
    self.assertTrue('*CCOC' in ks)