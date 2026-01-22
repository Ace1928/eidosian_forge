import json
from os import listdir, pathsep
from os.path import join, isfile, isdir, dirname
from subprocess import CalledProcessError
import contextlib
import platform
import itertools
import subprocess
import distutils.errors
from setuptools.extern.more_itertools import unique_everseen
def find_programdata_vs_vers(self):
    """
        Find Visual studio 2017+ versions from information in
        "C:\\ProgramData\\Microsoft\\VisualStudio\\Packages\\_Instances".

        Return
        ------
        dict
            float version as key, path as value.
        """
    vs_versions = {}
    instances_dir = 'C:\\ProgramData\\Microsoft\\VisualStudio\\Packages\\_Instances'
    try:
        hashed_names = listdir(instances_dir)
    except OSError:
        return vs_versions
    for name in hashed_names:
        try:
            state_path = join(instances_dir, name, 'state.json')
            with open(state_path, 'rt', encoding='utf-8') as state_file:
                state = json.load(state_file)
            vs_path = state['installationPath']
            listdir(join(vs_path, 'VC\\Tools\\MSVC'))
            vs_versions[self._as_float_version(state['installationVersion'])] = vs_path
        except (OSError, KeyError):
            continue
    return vs_versions