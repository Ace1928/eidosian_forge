from __future__ import annotations
import os.path
import re
import shutil
import tempfile
import zipfile
from glob import iglob
from ..bdist_wheel import bdist_wheel
from ..wheelfile import WheelFile
from . import WheelError
def egg2wheel(egg_path: str, dest_dir: str) -> None:
    filename = os.path.basename(egg_path)
    match = egg_info_re.match(filename)
    if not match:
        raise WheelError(f'Invalid egg file name: {filename}')
    egg_info = match.groupdict()
    dir = tempfile.mkdtemp(suffix='_e2w')
    if os.path.isfile(egg_path):
        with zipfile.ZipFile(egg_path) as egg:
            egg.extractall(dir)
    else:
        for pth in os.listdir(egg_path):
            src = os.path.join(egg_path, pth)
            if os.path.isfile(src):
                shutil.copy2(src, dir)
            else:
                shutil.copytree(src, os.path.join(dir, pth))
    pyver = egg_info['pyver']
    if pyver:
        pyver = egg_info['pyver'] = pyver.replace('.', '')
    arch = (egg_info['arch'] or 'any').replace('.', '_').replace('-', '_')
    abi = 'cp' + pyver[2:] if arch != 'any' else 'none'
    root_is_purelib = egg_info['arch'] is None
    if root_is_purelib:
        bw = bdist_wheel(Distribution())
    else:
        bw = _bdist_wheel_tag(Distribution())
    bw.root_is_pure = root_is_purelib
    bw.python_tag = pyver
    bw.plat_name_supplied = True
    bw.plat_name = egg_info['arch'] or 'any'
    if not root_is_purelib:
        bw.full_tag_supplied = True
        bw.full_tag = (pyver, abi, arch)
    dist_info_dir = os.path.join(dir, '{name}-{ver}.dist-info'.format(**egg_info))
    bw.egg2dist(os.path.join(dir, 'EGG-INFO'), dist_info_dir)
    bw.write_wheelfile(dist_info_dir, generator='egg2wheel')
    wheel_name = '{name}-{ver}-{pyver}-{}-{}.whl'.format(abi, arch, **egg_info)
    with WheelFile(os.path.join(dest_dir, wheel_name), 'w') as wf:
        wf.write_files(dir)
    shutil.rmtree(dir)