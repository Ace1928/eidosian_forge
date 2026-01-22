import warnings
import testtools
import fixtures
def test_capture_category(self):
    self.useFixture(fixtures.WarningsFilter())
    warnings.simplefilter('always')
    w = self.useFixture(fixtures.WarningsCapture())
    categories = [DeprecationWarning, Warning, UserWarning, SyntaxWarning, RuntimeWarning, UnicodeWarning, FutureWarning]
    for category in categories:
        warnings.warn('test', category)
    self.assertEqual(len(categories), len(w.captures))
    for i, category in enumerate(categories):
        c = w.captures[i]
        self.assertEqual(category, c.category)