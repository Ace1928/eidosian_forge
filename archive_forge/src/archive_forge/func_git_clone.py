import os
import shutil
import subprocess
import nox
def git_clone(session, git_url):
    session.run('git', 'clone', '--depth', '1', git_url, external=True)