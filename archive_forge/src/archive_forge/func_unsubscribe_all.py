import collections
from oslo_log import log as logging
from oslo_utils import reflection
from neutron_lib.callbacks import events
from neutron_lib.callbacks import exceptions
from neutron_lib.callbacks import priority_group
from neutron_lib.db import utils as db_utils
def unsubscribe_all(self, callback):
    """Unsubscribe callback for all events and all resources.


        :param callback: the callback.
        """
    callback_id = self._find(callback)
    if callback_id:
        for resource, resource_events in self._index[callback_id].items():
            for event in resource_events:
                self._del_callback(self._callbacks[resource][event], callback_id)
        del self._index[callback_id]