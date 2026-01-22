from sqlalchemy.testing.requirements import Requirements
from alembic import util
from alembic.util import sqla_compat
from ..testing import exclusions
@property
def computed_reflects_normally(self):
    return exclusions.only_if(exclusions.BooleanPredicate(sqla_compat.has_computed_reflection))