import os
import shutil
import tempfile
from io import BytesIO
from dulwich import porcelain
from ...repo import Repo
from .utils import CompatTestCase, run_git_or_fail
Tests related to patch compatibility with CGit.