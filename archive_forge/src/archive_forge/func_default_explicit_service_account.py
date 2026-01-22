import os
import pathlib
import subprocess
import shutil
import tempfile
from nox.command import which
import nox
@nox.session(python=PYTHON_VERSIONS_SYNC)
def default_explicit_service_account(session):
    session.env[EXPLICIT_CREDENTIALS_ENV] = SERVICE_ACCOUNT_FILE
    session.env[EXPECT_PROJECT_ENV] = '1'
    session.install(*TEST_DEPENDENCIES_SYNC)
    session.install(LIBRARY_DIR)
    default(session, 'system_tests_sync/test_default.py', 'system_tests_sync/test_id_token.py', *session.posargs)