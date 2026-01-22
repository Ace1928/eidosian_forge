import os
import pathlib
import subprocess
import shutil
import tempfile
from nox.command import which
import nox
@nox.session(python=PYTHON_VERSIONS_ASYNC)
def default_cloud_sdk_authorized_user_configured_project_async(session):
    configure_cloud_sdk(session, AUTHORIZED_USER_FILE, project=True)
    session.env[EXPECT_PROJECT_ENV] = '1'
    session.install(*TEST_DEPENDENCIES_SYNC + TEST_DEPENDENCIES_ASYNC)
    session.install(LIBRARY_DIR)
    default(session, 'system_tests_async/test_default.py', *session.posargs)