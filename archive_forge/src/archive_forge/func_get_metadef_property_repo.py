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
def get_metadef_property_repo(self, context):
    """Get the layered MetadefPropertyRepo model.

        This is where we construct the "the onion" by layering
        MetadefPropertyRepo models on top of each other, starting with
        the DB at the bottom.

        :param context: The RequestContext
        :returns: An MetadefPropertyRepo-like object
        """
    repo = glance.db.MetadefPropertyRepo(context, self.db_api)
    repo = glance.notifier.MetadefPropertyRepoProxy(repo, context, self.notifier)
    return repo