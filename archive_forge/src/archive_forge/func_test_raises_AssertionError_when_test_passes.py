from testtools import matchers
from testtools import testcase
from keystone.tests.unit import utils
def test_raises_AssertionError_when_test_passes(self):

    @utils.wip('waiting on bug #000000')
    def test():
        pass
    e = self.assertRaises(AssertionError, test)
    self.assertThat(str(e), matchers.Contains('#000000'))