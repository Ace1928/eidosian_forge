import glance_store
from glance.api import policy
from glance.api import property_protections
from glance.common import property_utils
from glance.common import store_utils
import glance.db
import glance.domain
import glance.location
import glance.notifier
import glance.quota
def get_metadef_property_factory(self, context):
    factory = glance.domain.MetadefPropertyFactory()
    factory = glance.notifier.MetadefPropertyFactoryProxy(factory, context, self.notifier)
    return factory