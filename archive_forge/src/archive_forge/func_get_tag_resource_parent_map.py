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
def get_tag_resource_parent_map():
    parent_map = {}
    for subclass in HasStandardAttributes.__subclasses__():
        if subclass.validate_tag_support():
            for collection, resource in subclass.get_collection_resource_map().items():
                if collection in parent_map:
                    msg = _('API parent %(collection)s/%(resource)s for model %(subclass)s is already registered.') % {'collection': collection, 'resource': resource, 'subclass': subclass}
                    raise RuntimeError(msg)
                parent_map[collection] = resource
    return parent_map