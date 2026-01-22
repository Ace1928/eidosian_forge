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
def get_metadef_object_repo(self, context):
    """Get the layered MetadefObjectRepo model.

        This is where we construct the "the onion" by layering
        MetadefObjectRepo models on top of each other, starting with the DB at
        the bottom.

        :param context: The RequestContext
        :returns: An MetadefObjectRepo-like object
        """
    repo = glance.db.MetadefObjectRepo(context, self.db_api)
    repo = glance.notifier.MetadefObjectRepoProxy(repo, context, self.notifier)
    return repo