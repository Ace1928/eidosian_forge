import os
import shutil
import subprocess
import sys
from pathlib import Path
def build_all():
    assert build() == 'ok'
    tar = next(Path('/tmp/ase-docs--ok/ase/dist/').glob('ase-*.tar.gz'))
    webpage = Path('/tmp/ase-docs-ok/ase/doc/ase-web-page')
    home = Path.home() / 'web-pages'
    cmds = ' && '.join([f'cp {tar} {webpage}', f'find {webpage} -name install.html | xargs sed -i s/snapshot.tar.gz/{tar.name}/g', f'cd {webpage.parent}', 'tar -czf ase-web-page.tar.gz ase-web-page', f'cp ase-web-page.tar.gz {home}'])
    subprocess.run(cmds, shell=True, check=True)