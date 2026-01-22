from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
def test_skip2(self):
    raise RuntimeError('Ought to skip me')