from ...revision import Revision
from ...tests import TestCase, TestCaseWithTransport
from .cmds import collapse_by_person, get_revisions_and_committers
def test_no_conflicts(self):
    revisions = [Revision('1', {}, committer='Foo <foo@example.com>'), Revision('2', {}, committer='Bar <bar@example.com>'), Revision('3', {}, committer='Bar <bar@example.com>')]
    foo = ('Foo', 'foo@example.com')
    bar = ('Bar', 'bar@example.com')
    committers = {foo: foo, bar: bar}
    info = collapse_by_person(revisions, committers)
    self.assertEqual(2, info[0][0])
    self.assertEqual({'bar@example.com': 2}, info[0][2])
    self.assertEqual({'Bar': 2}, info[0][3])