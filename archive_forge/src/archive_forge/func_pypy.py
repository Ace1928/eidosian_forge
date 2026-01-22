import os
import pathlib
import shutil
import nox
@nox.session(python='pypy')
def pypy(session):
    session.install('-r', 'test/requirements.txt')
    session.install('-e', '.')
    session.run('pytest', f'--junitxml=unit_{session.python}_sponge_log.xml', '--cov=google.auth', '--cov=google.oauth2', '--cov=tests', 'tests', 'tests_async')