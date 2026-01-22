from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import scheduler_hints as sh
from heat.engine.resources import volume_base as vb
from heat.engine import support
from heat.engine import translation
def handle_restore(self, defn, restore_data):
    backup_id = restore_data['resource_data']['backup_id']
    ignore_props = (self.IMAGE_REF, self.IMAGE, self.SOURCE_VOLID)
    props = dict(((key, value) for key, value in self.properties.data.items() if key not in ignore_props and value is not None))
    props[self.BACKUP_ID] = backup_id
    return defn.freeze(properties=props)