import sys
import types
import eventlet
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from heat.common.i18n import _
from heat.common import timeutils
def cancel_all(self, grace_period=None):
    if callable(grace_period):
        get_grace_period = grace_period
    else:

        def get_grace_period(key):
            return grace_period
    for k, r in self._runners.items():
        if not r.started() or r.done():
            gp = None
        else:
            gp = get_grace_period(k)
        try:
            r.cancel(grace_period=gp)
        except Exception as ex:
            LOG.debug('Exception cancelling task: %s', str(ex))