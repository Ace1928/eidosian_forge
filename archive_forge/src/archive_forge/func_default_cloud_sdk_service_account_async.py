import os
import pathlib
import subprocess
import shutil
import tempfile
from nox.command import which
import nox
@nox.session(python=PYTHON_VERSIONS_ASYNC)
def default_cloud_sdk_service_account_async(session):
    configure_cloud_sdk(session, SERVICE_ACCOUNT_FILE)
    session.env[EXPECT_PROJECT_ENV] = '1'
    session.install(*TEST_DEPENDENCIES_SYNC + TEST_DEPENDENCIES_ASYNC)
    session.install(LIBRARY_DIR)
    default(session, 'system_tests_async/test_default.py', *session.posargs)