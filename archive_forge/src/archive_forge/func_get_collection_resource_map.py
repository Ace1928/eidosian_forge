from oslo_utils import timeutils
import sqlalchemy as sa
from sqlalchemy import event  # noqa
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext import declarative
from sqlalchemy.orm import attributes
from sqlalchemy.orm import session as se
from neutron_lib._i18n import _
from neutron_lib.db import constants as db_const
from neutron_lib.db import model_base
from neutron_lib.db import sqlalchemytypes
@classmethod
def get_collection_resource_map(cls):
    try:
        return cls.collection_resource_map
    except AttributeError as e:
        raise NotImplementedError(_('%s must define collection_resource_map') % cls) from e