import copy
import re
import sys
from rdkit import Chem
from rdkit import RDRandom as random
from rdkit.Chem import rdChemReactions as Reactions
def testGithub1734(self):
    m = Chem.MolFromSmiles('c1ccccc1[C@H](C)NC')
    res = BRICSDecompose(m)
    self.assertEqual(len(res), 3)
    self.assertTrue('[4*][C@H]([8*])C' in res)
    res = BreakBRICSBonds(m)
    self.assertEqual(Chem.MolToSmiles(res, isomericSmiles=True), '[16*]c1ccccc1.[4*][C@H]([8*])C.[5*]NC')