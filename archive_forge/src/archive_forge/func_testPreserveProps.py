import unittest
from rdkit import Chem
from rdkit.Chem.MolStandardize.standardize import Standardizer
def testPreserveProps(self):
    PROP_NAME = 'MyProp'
    PROP_VALUE = 'foo'
    standardizer = FakeStandardizer()
    m = Chem.MolFromSmiles('C')
    m.SetProp(PROP_NAME, PROP_VALUE)
    standardized_mol = standardizer.standardize(m)
    self.assertTrue(standardized_mol.HasProp(PROP_NAME))
    self.assertEqual(PROP_VALUE, standardized_mol.GetProp(PROP_NAME))