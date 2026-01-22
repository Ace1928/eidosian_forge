import os
from alembic import config as alembic_config
from alembic import migration as alembic_migration
from alembic import script as alembic_script
from sqlalchemy import MetaData, Table
from glance.db.sqlalchemy import api as db_api
def update_alembic_version(old, new):
    """Correct alembic head in order to upgrade DB using EMC method.

            :param:old: Actual alembic head
            :param:new: Expected alembic head to be updated
            """
    meta = MetaData()
    alembic_version = Table('alembic_version', meta, autoload_with=engine)
    alembic_version.update().values(version_num=new).where(alembic_version.c.version_num == old).execute()