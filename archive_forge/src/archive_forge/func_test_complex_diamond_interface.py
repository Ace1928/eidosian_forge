import unittest
def test_complex_diamond_interface(self):
    from zope.interface import Interface
    IA = self._make_complex_diamond(Interface)
    self.assertEqual([x.__name__ for x in IA.__iro__], ['A', 'B', 'C', 'D', 'E', 'F', 'Interface'])