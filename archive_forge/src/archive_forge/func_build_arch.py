import shutil
import zipfile
from pathlib import Path
from pythonforandroid.logger import info
from pythonforandroid.recipe import PythonRecipe
def build_arch(self, arch):
    """ Unzip the wheel and copy into site-packages of target"""
    info('Installing {} into site-packages'.format(self.name))
    with zipfile.ZipFile(self.wheel_path, 'r') as zip_ref:
        info('Unzip wheels and copy into {}'.format(self.ctx.get_python_install_dir(arch.arch)))
        zip_ref.extractall(self.ctx.get_python_install_dir(arch.arch))
    lib_dir = Path(f'{self.ctx.get_python_install_dir(arch.arch)}/shiboken6')
    shutil.copyfile(lib_dir / 'libshiboken6.abi3.so', Path(self.ctx.get_libs_dir(arch.arch)) / 'libshiboken6.abi3.so')