from Cython.Build.Dependencies import strip_string_literals
from Cython.TestUtils import CythonTest
def test_comment_and_quote(self):
    self.t("abc # 'x'", 'abc #_L1_')
    self.t("'abc#'", "'_L1_'")