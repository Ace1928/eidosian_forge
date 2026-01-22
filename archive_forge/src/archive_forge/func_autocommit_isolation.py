from sqlalchemy.testing.requirements import Requirements
from alembic import util
from alembic.util import sqla_compat
from ..testing import exclusions
@property
def autocommit_isolation(self):
    """target database should support 'AUTOCOMMIT' isolation level"""
    return exclusions.closed()