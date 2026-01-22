import datetime
import sqlalchemy
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import orm
from sqlalchemy.orm import collections
from keystone.common import password_hashing
from keystone.common import resource_options
from keystone.common import sql
import keystone.conf
from keystone.identity.backends import resource_options as iro
@property
def password_is_expired(self):
    """Return whether password is expired or not."""
    if self.password_expires_at and (not self._password_expiry_exempt()):
        return datetime.datetime.utcnow() >= self.password_expires_at
    return False