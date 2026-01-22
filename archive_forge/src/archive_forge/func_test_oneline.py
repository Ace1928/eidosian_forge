import io
from .. import errors, i18n, tests, workingtree
def test_oneline(self):
    self.assertEqual('zzå{{spam ham eggs}}', i18n.gettext_per_paragraph('spam ham eggs'))