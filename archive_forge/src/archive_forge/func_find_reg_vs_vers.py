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
def find_reg_vs_vers(self):
    """
        Find Microsoft Visual Studio versions available in registry.

        Return
        ------
        list of float
            Versions
        """
    ms = self.ri.microsoft
    vckeys = (self.ri.vc, self.ri.vc_for_python, self.ri.vs)
    vs_vers = []
    for hkey, key in itertools.product(self.ri.HKEYS, vckeys):
        try:
            bkey = winreg.OpenKey(hkey, ms(key), 0, winreg.KEY_READ)
        except OSError:
            continue
        with bkey:
            subkeys, values, _ = winreg.QueryInfoKey(bkey)
            for i in range(values):
                with contextlib.suppress(ValueError):
                    ver = float(winreg.EnumValue(bkey, i)[0])
                    if ver not in vs_vers:
                        vs_vers.append(ver)
            for i in range(subkeys):
                with contextlib.suppress(ValueError):
                    ver = float(winreg.EnumKey(bkey, i))
                    if ver not in vs_vers:
                        vs_vers.append(ver)
    return sorted(vs_vers)