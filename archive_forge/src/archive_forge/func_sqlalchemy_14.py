from sqlalchemy.testing.requirements import Requirements
from alembic import util
from alembic.util import sqla_compat
from ..testing import exclusions
@property
def sqlalchemy_14(self):
    return exclusions.skip_if(lambda config: not util.sqla_14, 'SQLAlchemy 1.4 or greater required')