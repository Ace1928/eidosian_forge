from ... import tests
from .. import generate_ids
def test_gen_revision_id_user(self):
    """If there is no email, fall back to the whole username"""
    tail = b'-\\d{14}-[a-z0-9]{16}'
    self.assertGenRevisionId(b'joe_bar' + tail, 'Joe Bar')
    self.assertGenRevisionId(b'joebar' + tail, 'joebar')
    self.assertGenRevisionId(b'joe_br' + tail, 'Joe Bår')
    self.assertGenRevisionId(b'joe_br_user\\+joe_bar_foo-bar.com' + tail, 'Joe Bår <user+Joe_Bar_Foo-Bar.com>')