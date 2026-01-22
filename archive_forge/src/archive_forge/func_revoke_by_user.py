from keystone.common import cache
from keystone.common import manager
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.models import revoke_model
from keystone import notifications
def revoke_by_user(self, user_id):
    return self.revoke(revoke_model.RevokeEvent(user_id=user_id))