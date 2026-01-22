from ... import tests
from .. import generate_ids
def test_gen_revision_id_email(self):
    """gen_revision_id uses email address if present"""
    regex = b'user\\+joe_bar@foo-bar\\.com-\\d{14}-[a-z0-9]{16}'
    self.assertGenRevisionId(regex, 'user+joe_bar@foo-bar.com')
    self.assertGenRevisionId(regex, '<user+joe_bar@foo-bar.com>')
    self.assertGenRevisionId(regex, 'Joe Bar <user+joe_bar@foo-bar.com>')
    self.assertGenRevisionId(regex, 'Joe Bar <user+Joe_Bar@Foo-Bar.com>')
    self.assertGenRevisionId(regex, 'Joe Bår <user+Joe_Bar@Foo-Bar.com>')