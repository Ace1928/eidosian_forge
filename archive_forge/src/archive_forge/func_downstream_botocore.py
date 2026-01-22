import os
import shutil
import subprocess
import nox
@nox.session(python=['3.9'])
def downstream_botocore(session):
    root = os.getcwd()
    tmp_dir = session.create_tmp()
    session.cd(tmp_dir)
    git_clone(session, 'https://github.com/boto/botocore')
    session.chdir('botocore')
    session.run('git', 'rev-parse', 'HEAD', external=True)
    session.run('python', 'scripts/ci/install')
    session.cd(root)
    session.install('.', silent=False)
    session.cd(f'{tmp_dir}/botocore')
    session.run('python', 'scripts/ci/run-tests')