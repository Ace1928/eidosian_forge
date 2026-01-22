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
Override from_dict to remove password_expires_at attribute.

        Overriding this method to remove password_expires_at attribute to
        support update_user and unit tests where password_expires_at
        inadvertently gets added by calling to_dict followed by from_dict.

        :param user_dict: User entity dictionary
        :returns User: User object

        