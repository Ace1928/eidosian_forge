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
def make_sourceless(path, style):
    import py_compile
    py_compile.compile(path)
    if style == 'simple':
        pyc_path = util.pyc_file_from_path(path)
        suffix = importlib.machinery.BYTECODE_SUFFIXES[0]
        filepath, ext = os.path.splitext(path)
        simple_pyc_path = filepath + suffix
        shutil.move(pyc_path, simple_pyc_path)
        pyc_path = simple_pyc_path
    else:
        assert style in ('pep3147', 'simple')
        pyc_path = util.pyc_file_from_path(path)
    assert os.access(pyc_path, os.F_OK)
    os.unlink(path)