from sqlalchemy.testing.requirements import Requirements
from alembic import util
from alembic.util import sqla_compat
from ..testing import exclusions
@property
def autoincrement_on_composite_pk(self):
    return exclusions.closed()