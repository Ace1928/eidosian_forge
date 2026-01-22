import os
import pathlib
import subprocess
import shutil
import tempfile
from nox.command import which
import nox
@nox.session(python=PYTHON_VERSIONS_SYNC)
def compute_engine(session):
    session.install(*TEST_DEPENDENCIES_SYNC)
    del session.virtualenv.env['GOOGLE_APPLICATION_CREDENTIALS']
    session.install(LIBRARY_DIR)
    default(session, 'system_tests_sync/test_compute_engine.py', *session.posargs)