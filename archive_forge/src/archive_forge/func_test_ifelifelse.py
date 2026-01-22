from Cython.TestUtils import CythonTest
def test_ifelifelse(self):
    self.t(u'\n                    if x:\n                        pass\n                    elif y:\n                        pass\n                    elif z + 34 ** 34 - 2:\n                        pass\n                    else:\n                        pass\n                ')