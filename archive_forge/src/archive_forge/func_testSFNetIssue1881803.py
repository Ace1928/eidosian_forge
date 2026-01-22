import sys
import weakref
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions
def testSFNetIssue1881803(self):
    m = Chem.MolFromSmiles('c1ccccc1n1cccc1')
    res = RecapDecompose(m)
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 2)
    m = Chem.MolFromSmiles('c1ccccc1[n+]1ccccc1')
    res = RecapDecompose(m)
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 0)
    m = Chem.MolFromSmiles('C1CC1NC(=O)CC')
    res = RecapDecompose(m)
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 2)
    m = Chem.MolFromSmiles('C1CC1[NH+]C(=O)CC')
    res = RecapDecompose(m)
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 0)
    m = Chem.MolFromSmiles('C1CC1NC(=O)NC1CCC1')
    res = RecapDecompose(m)
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 2)
    m = Chem.MolFromSmiles('C1CC1[NH+]C(=O)[NH+]C1CCC1')
    res = RecapDecompose(m)
    self.assertTrue(res)
    self.assertTrue(len(res.GetLeaves()) == 0)