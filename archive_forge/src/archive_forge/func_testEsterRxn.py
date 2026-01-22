import sys
import weakref
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions
def testEsterRxn(self):
    m = Chem.MolFromSmiles('C1CC1C(=O)OC1OC1')
    res = RecapDecompose(m, onlyUseReactions=[2])
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 2)
    ks = res.GetLeaves().keys()
    self.assertTrue('*C(=O)C1CC1' in ks)
    self.assertTrue('*OC1CO1' in ks)
    m = Chem.MolFromSmiles('C1CC1C(=O)CC1OC1')
    res = RecapDecompose(m, onlyUseReactions=[2])
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 0)
    m = Chem.MolFromSmiles('C1CCC(=O)OC1')
    res = RecapDecompose(m, onlyUseReactions=[2])
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 0)