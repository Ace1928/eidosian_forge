import os
import pathlib
import subprocess
import shutil
import tempfile
from nox.command import which
import nox
@nox.session(python=PYTHON_VERSIONS_SYNC)
def default_cloud_sdk_authorized_user_configured_project(session):
    configure_cloud_sdk(session, AUTHORIZED_USER_FILE, project=True)
    session.env[EXPECT_PROJECT_ENV] = '1'
    session.install(*TEST_DEPENDENCIES_SYNC)
    session.install(LIBRARY_DIR)
    default(session, 'system_tests_sync/test_default.py', *session.posargs)