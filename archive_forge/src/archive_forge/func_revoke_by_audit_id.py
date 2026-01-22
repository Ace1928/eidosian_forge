from keystone.common import cache
from keystone.common import manager
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.models import revoke_model
from keystone import notifications
def revoke_by_audit_id(self, audit_id):
    self.revoke(revoke_model.RevokeEvent(audit_id=audit_id))