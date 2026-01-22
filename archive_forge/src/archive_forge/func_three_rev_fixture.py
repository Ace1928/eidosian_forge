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
def three_rev_fixture(cfg):
    a = util.rev_id()
    b = util.rev_id()
    c = util.rev_id()
    script = ScriptDirectory.from_config(cfg)
    script.generate_revision(a, 'revision a', refresh=True, head='base')
    write_script(script, a, '"Rev A"\nrevision = \'%s\'\ndown_revision = None\n\nfrom alembic import op\n\n\ndef upgrade():\n    op.execute("CREATE STEP 1")\n\n\ndef downgrade():\n    op.execute("DROP STEP 1")\n\n' % a)
    script.generate_revision(b, 'revision b', refresh=True, head=a)
    write_script(script, b, f"""# coding: utf-8\n"Rev B, méil, %3"\nrevision = '{b}'\ndown_revision = '{a}'\n\nfrom alembic import op\n\n\ndef upgrade():\n    op.execute("CREATE STEP 2")\n\n\ndef downgrade():\n    op.execute("DROP STEP 2")\n\n""", encoding='utf-8')
    script.generate_revision(c, 'revision c', refresh=True, head=b)
    write_script(script, c, '"Rev C"\nrevision = \'%s\'\ndown_revision = \'%s\'\n\nfrom alembic import op\n\n\ndef upgrade():\n    op.execute("CREATE STEP 3")\n\n\ndef downgrade():\n    op.execute("DROP STEP 3")\n\n' % (c, b))
    return (a, b, c)