from oslo_versionedobjects import base
from oslo_versionedobjects import fields
from heat.db import api as db_api
from heat.objects import base as heat_base
Method to help with migration to objects.

        Converts a database entity to a formal object.
        