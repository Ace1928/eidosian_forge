import datetime
import eventlet
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from oslo_log import log as logging
def simulated_effort():
    client_name = self.properties[self.CLIENT_NAME]
    self.entity = self.properties[self.ENTITY_NAME]
    if client_name and self.entity:
        entity_id = self.data().get('value') or self.resource_id
        try:
            obj = getattr(self.client(name=client_name), self.entity)
            obj.get(entity_id)
        except Exception as exc:
            LOG.debug('%s.%s(%s) %s' % (client_name, self.entity, entity_id, str(exc)))
    else:
        eventlet.sleep(1)