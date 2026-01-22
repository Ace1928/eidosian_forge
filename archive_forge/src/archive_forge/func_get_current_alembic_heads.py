import os
from alembic import config as alembic_config
from alembic import migration as alembic_migration
from alembic import script as alembic_script
from sqlalchemy import MetaData, Table
from glance.db.sqlalchemy import api as db_api
def get_current_alembic_heads():
    """Return current heads (if any) from the alembic migration table"""
    engine = db_api.get_engine()
    with engine.connect() as conn:
        context = alembic_migration.MigrationContext.configure(conn)
        heads = context.get_current_heads()

        def update_alembic_version(old, new):
            """Correct alembic head in order to upgrade DB using EMC method.

            :param:old: Actual alembic head
            :param:new: Expected alembic head to be updated
            """
            meta = MetaData()
            alembic_version = Table('alembic_version', meta, autoload_with=engine)
            alembic_version.update().values(version_num=new).where(alembic_version.c.version_num == old).execute()
        if 'pike01' in heads:
            update_alembic_version('pike01', 'pike_contract01')
        elif 'ocata01' in heads:
            update_alembic_version('ocata01', 'ocata_contract01')
        heads = context.get_current_heads()
        return heads