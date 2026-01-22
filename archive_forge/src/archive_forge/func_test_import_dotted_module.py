import unittest
from traits.util.api import import_symbol
def test_import_dotted_module(self):
    """ import dotted module """
    symbol = import_symbol('traits.util.import_symbol:import_symbol')
    self.assertEqual(symbol, import_symbol)