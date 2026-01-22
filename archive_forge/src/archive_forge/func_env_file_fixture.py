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
def env_file_fixture(txt):
    dir_ = os.path.join(_get_staging_directory(), 'scripts')
    txt = '\nfrom alembic import context\n\nconfig = context.config\n' + txt
    path = os.path.join(dir_, 'env.py')
    pyc_path = util.pyc_file_from_path(path)
    if pyc_path:
        os.unlink(pyc_path)
    with open(path, 'w') as f:
        f.write(txt)