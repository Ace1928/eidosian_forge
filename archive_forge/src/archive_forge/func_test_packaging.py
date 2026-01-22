import pytest
import os
import subprocess
import sys
import shutil
def test_packaging(self):
    dist = os.path.join(self.pinstall_path, 'dist')
    build = os.path.join(self.pinstall_path, 'build')
    try:
        subprocess.check_output([sys.executable or 'python', '-m', 'PyInstaller', os.path.join(self.pinstall_path, 'main.spec'), '--distpath', dist, '--workpath', build], stderr=subprocess.STDOUT, env=self.env)
    except subprocess.CalledProcessError as e:
        print(e.output.decode('utf8'))
        raise