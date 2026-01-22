import os
import routes
import webob
from glance.api.middleware import context
from glance.api.v2 import router
import glance.common.client
def fake_sendable(self, body):
    force = getattr(self, 'stub_force_sendfile', None)
    if force is None:
        return self._stub_orig_sendable(body)
    else:
        if force:
            assert glance.common.client.SENDFILE_SUPPORTED
        return force