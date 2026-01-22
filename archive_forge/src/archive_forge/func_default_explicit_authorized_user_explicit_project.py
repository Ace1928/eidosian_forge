import os
import pathlib
import subprocess
import shutil
import tempfile
from nox.command import which
import nox
@nox.session(python=PYTHON_VERSIONS_SYNC)
def default_explicit_authorized_user_explicit_project(session):
    session.env[EXPLICIT_CREDENTIALS_ENV] = AUTHORIZED_USER_FILE
    session.env[EXPLICIT_PROJECT_ENV] = 'example-project'
    session.env[EXPECT_PROJECT_ENV] = '1'
    session.install(*TEST_DEPENDENCIES_SYNC)
    session.install(LIBRARY_DIR)
    default(session, 'system_tests_sync/test_default.py', *session.posargs)