import testtools
def deassociate(self, *args):
    resp = self.controller.deassociate(*args)
    self._assertRequestId(resp)