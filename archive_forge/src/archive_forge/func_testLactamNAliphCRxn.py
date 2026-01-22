import sys
import weakref
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions
def testLactamNAliphCRxn(self):
    m = Chem.MolFromSmiles('C1CC(=O)N1CCCC')
    res = RecapDecompose(m, onlyUseReactions=[8])
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 2)
    ks = res.GetLeaves().keys()
    self.assertTrue('*N1CCC1=O' in ks)
    self.assertTrue('*CCCC' in ks)
    m = Chem.MolFromSmiles('O=C1CC2N1CCCC2')
    res = RecapDecompose(m)
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 0)