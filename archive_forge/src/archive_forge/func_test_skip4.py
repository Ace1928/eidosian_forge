from twisted.trial.unittest import FailTest, SkipTest, SynchronousTestCase, TestCase
def test_skip4(self):
    raise RuntimeError('Skip me too')