import os
import pathlib
import subprocess
import shutil
import tempfile
from nox.command import which
import nox
def install_cloud_sdk(session):
    """Downloads and installs the Google Cloud SDK."""
    session.env[CLOUD_SDK_CONFIG_ENV] = str(CLOUD_SDK_ROOT)
    session.env[CLOUD_SDK_PYTHON_ENV] = CLOUD_SDK_PYTHON
    session.env['PATH'] = str(CLOUD_SDK_INSTALL_DIR.joinpath('bin')) + os.pathsep + os.environ['PATH']
    if pathlib.Path(GCLOUD).exists():
        session.run(GCLOUD, 'components', 'update', '-q')
        return
    tar_path = CLOUD_SDK_ROOT.joinpath(CLOUD_SDK_DIST_FILENAME)
    session.run('wget', CLOUD_SDK_DOWNLOAD_URL, '-O', str(tar_path), silent=True)
    session.run('tar', 'xzf', str(tar_path), '-C', str(CLOUD_SDK_ROOT))
    tar_path.unlink()
    session.run(str(CLOUD_SDK_INSTALL_DIR.joinpath('install.sh')), '--usage-reporting', 'false', '--path-update', 'false', '--command-completion', 'false', silent=True)