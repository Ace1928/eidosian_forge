import importlib.machinery
import os
import shutil
import textwrap
from sqlalchemy.testing import config
from sqlalchemy.testing import provision
from . import util as testing_util
from .. import command
from .. import script
from .. import util
from ..script import Script
from ..script import ScriptDirectory
from alembic import context
from alembic import op
from alembic import op
from alembic import op
from alembic import op
from alembic import op
from alembic import op
def multi_heads_fixture(cfg, a, b, c):
    """Create a multiple head fixture from the three-revs fixture"""
    d = util.rev_id()
    e = util.rev_id()
    f = util.rev_id()
    script = ScriptDirectory.from_config(cfg)
    script.generate_revision(d, 'revision d from b', head=b, splice=True, refresh=True)
    write_script(script, d, '"Rev D"\nrevision = \'%s\'\ndown_revision = \'%s\'\n\nfrom alembic import op\n\n\ndef upgrade():\n    op.execute("CREATE STEP 4")\n\n\ndef downgrade():\n    op.execute("DROP STEP 4")\n\n' % (d, b))
    script.generate_revision(e, 'revision e from d', head=d, splice=True, refresh=True)
    write_script(script, e, '"Rev E"\nrevision = \'%s\'\ndown_revision = \'%s\'\n\nfrom alembic import op\n\n\ndef upgrade():\n    op.execute("CREATE STEP 5")\n\n\ndef downgrade():\n    op.execute("DROP STEP 5")\n\n' % (e, d))
    script.generate_revision(f, 'revision f from b', head=b, splice=True, refresh=True)
    write_script(script, f, '"Rev F"\nrevision = \'%s\'\ndown_revision = \'%s\'\n\nfrom alembic import op\n\n\ndef upgrade():\n    op.execute("CREATE STEP 6")\n\n\ndef downgrade():\n    op.execute("DROP STEP 6")\n\n' % (f, b))
    return (d, e, f)