from contextlib import contextmanager
import glob
from importlib import import_module
import io
import itertools
import os.path as osp
import re
import sys
import warnings
import zipfile
import configparser
def iter_files_distros(path=None, repeated_distro='first'):
    if path is None:
        path = sys.path
    distro_names_seen = set()
    for folder in path:
        if folder.rstrip('/\\').endswith('.egg'):
            egg_name = osp.basename(folder)
            distro = Distribution.from_name_version(egg_name.split('.egg')[0])
            if repeated_distro == 'first' and distro.name in distro_names_seen:
                continue
            distro_names_seen.add(distro.name)
            if osp.isdir(folder):
                ep_path = osp.join(folder, 'EGG-INFO', 'entry_points.txt')
                if osp.isfile(ep_path):
                    cp = CaseSensitiveConfigParser(delimiters=('=',))
                    cp.read([ep_path])
                    yield (cp, distro)
            elif zipfile.is_zipfile(folder):
                z = zipfile.ZipFile(folder)
                try:
                    info = z.getinfo('EGG-INFO/entry_points.txt')
                except KeyError:
                    continue
                cp = CaseSensitiveConfigParser(delimiters=('=',))
                with z.open(info) as f:
                    fu = io.TextIOWrapper(f)
                    cp.read_file(fu, source=osp.join(folder, 'EGG-INFO', 'entry_points.txt'))
                yield (cp, distro)
        elif zipfile.is_zipfile(folder):
            with zipfile.ZipFile(folder) as zf:
                for info in zf.infolist():
                    m = file_in_zip_pattern.match(info.filename)
                    if not m:
                        continue
                    distro_name_version = m.group('dist_version')
                    distro = Distribution.from_name_version(distro_name_version)
                    if repeated_distro == 'first' and distro.name in distro_names_seen:
                        continue
                    distro_names_seen.add(distro.name)
                    cp = CaseSensitiveConfigParser(delimiters=('=',))
                    with zf.open(info) as f:
                        fu = io.TextIOWrapper(f)
                        cp.read_file(fu, source=osp.join(folder, info.filename))
                    yield (cp, distro)
        for path in itertools.chain(glob.iglob(osp.join(glob.escape(folder), '*.dist-info', 'entry_points.txt')), glob.iglob(osp.join(glob.escape(folder), '*.egg-info', 'entry_points.txt'))):
            distro_name_version = osp.splitext(osp.basename(osp.dirname(path)))[0]
            distro = Distribution.from_name_version(distro_name_version)
            if repeated_distro == 'first' and distro.name in distro_names_seen:
                continue
            distro_names_seen.add(distro.name)
            cp = CaseSensitiveConfigParser(delimiters=('=',))
            cp.read([path])
            yield (cp, distro)