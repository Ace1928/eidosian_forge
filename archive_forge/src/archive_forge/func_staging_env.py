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
def staging_env(create=True, template='generic', sourceless=False):
    cfg = _testing_config()
    if create:
        path = os.path.join(_get_staging_directory(), 'scripts')
        assert not os.path.exists(path), 'staging directory %s already exists; poor cleanup?' % path
        command.init(cfg, path, template=template)
        if sourceless:
            try:
                util.load_python_file(path, 'env.py')
            except AttributeError:
                pass
            assert sourceless in ('pep3147_envonly', 'simple', 'pep3147_everything'), sourceless
            make_sourceless(os.path.join(path, 'env.py'), 'pep3147' if 'pep3147' in sourceless else 'simple')
    sc = script.ScriptDirectory.from_config(cfg)
    return sc