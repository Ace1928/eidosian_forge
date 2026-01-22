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
def get_task_executor_factory(self, context, admin_context=None):
    task_repo = self.get_task_repo(context)
    image_repo = self.get_repo(context)
    image_factory = self.get_image_factory(context)
    if admin_context:
        admin_repo = self.get_repo(admin_context)
    else:
        admin_repo = None
    return glance.domain.TaskExecutorFactory(task_repo, image_repo, image_factory, admin_repo=admin_repo)