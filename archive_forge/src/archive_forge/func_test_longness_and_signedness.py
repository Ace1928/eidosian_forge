from Cython.TestUtils import CythonTest
def test_longness_and_signedness(self):
    self.t(u'def f(unsigned long long long long long int y):\n    pass')