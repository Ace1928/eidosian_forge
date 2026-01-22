import unittest
from traits.util.api import import_symbol
def test_import_nested_symbol(self):
    """ import nested symbol """
    import tarfile
    symbol = import_symbol('tarfile:TarFile.open')
    self.assertEqual(symbol, tarfile.TarFile.open)