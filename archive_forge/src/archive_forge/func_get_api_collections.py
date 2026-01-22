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
def get_api_collections(cls):
    """Define the API collection this object will appear under.

        This should return a list of API collections that the object
        will be exposed under. Most should be exposed in just one
        collection (e.g. the network model is just exposed under
        'networks').

        This is used by the standard attr extensions to discover which
        resources need to be extended with the standard attr fields
        (e.g. created_at/updated_at/etc).
        """
    if hasattr(cls, 'api_collections'):
        return cls.api_collections
    raise NotImplementedError(_('%s must define api_collections') % cls)