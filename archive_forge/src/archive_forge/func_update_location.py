import testtools
def update_location(self, *args, **kwargs):
    resp = self.controller.update_location(*args, **kwargs)
    self._assertRequestId(resp)