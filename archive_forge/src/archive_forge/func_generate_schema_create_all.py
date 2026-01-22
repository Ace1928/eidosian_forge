import fixtures
import os
import testresources
import testscenarios
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
def generate_schema_create_all(self, engine):
    """A hook which should generate the model schema using create_all().

        This hook is called within the scope of creating the database
        assuming BUILD_WITH_MIGRATIONS is False.

        """