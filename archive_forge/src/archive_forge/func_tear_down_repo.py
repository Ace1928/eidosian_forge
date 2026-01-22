import datetime
import os
import shutil
import tempfile
import time
import types
import warnings
from dulwich.tests import SkipTest
from ..index import commit_tree
from ..objects import Commit, FixedSha, Tag, object_class
from ..pack import (
from ..repo import Repo
def tear_down_repo(repo):
    """Tear down a test repository."""
    repo.close()
    temp_dir = os.path.dirname(repo.path.rstrip(os.sep))
    shutil.rmtree(temp_dir)