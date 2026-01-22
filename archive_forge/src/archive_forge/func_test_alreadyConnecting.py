from twisted.internet import defer, error
from twisted.trial import unittest
from twisted.words.im import basesupport
def test_alreadyConnecting(self):
    """
        Test that it can fail sensibly when someone tried to connect before
        we did.
        """
    account = self.makeAccount()
    ui = self.makeUI()
    account.logOn(ui)
    self.assertRaises(error.ConnectError, account.logOn, ui)