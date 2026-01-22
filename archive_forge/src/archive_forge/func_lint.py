import os
import shutil
import subprocess
import nox
@nox.session
def lint(session):
    session.install('pre-commit')
    session.run('pre-commit', 'run', '--all-files')