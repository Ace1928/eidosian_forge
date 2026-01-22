from Cython.Build.Dependencies import strip_string_literals
from Cython.TestUtils import CythonTest
def test_raw_unicode(self):
    self.t("ru'abc\\\\'", "ru'_L1_'")