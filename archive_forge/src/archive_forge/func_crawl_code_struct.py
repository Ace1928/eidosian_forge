import xml.dom.minidom
import subprocess
import os
from shutil import rmtree
import keyword
from ..base import (CommandLine, CommandLineInputSpec, SEMLikeCommandLine, TraitedSpec,
import os\n\n\n"""
def crawl_code_struct(code_struct, package_dir):
    subpackages = []
    for k, v in code_struct.items():
        if isinstance(v, str) or isinstance(v, (str, bytes)):
            module_name = k.lower()
            class_name = k
            class_code = v
            add_class_to_package([class_code], [class_name], module_name, package_dir)
        else:
            l1 = {}
            l2 = {}
            for key in list(v.keys()):
                if isinstance(v[key], str) or isinstance(v[key], (str, bytes)):
                    l1[key] = v[key]
                else:
                    l2[key] = v[key]
            if l2:
                v = l2
                subpackages.append(k.lower())
                f_i = open(os.path.join(package_dir, '__init__.py'), 'a+')
                f_i.write('from %s import *\n' % k.lower())
                f_i.close()
                new_pkg_dir = os.path.join(package_dir, k.lower())
                if os.path.exists(new_pkg_dir):
                    rmtree(new_pkg_dir)
                os.mkdir(new_pkg_dir)
                crawl_code_struct(v, new_pkg_dir)
                if l1:
                    for ik, iv in l1.items():
                        crawl_code_struct({ik: {ik: iv}}, new_pkg_dir)
            elif l1:
                v = l1
                module_name = k.lower()
                add_class_to_package(list(v.values()), list(v.keys()), module_name, package_dir)
        if subpackages:
            f = open(os.path.join(package_dir, 'setup.py'), 'w')
            f.write("# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-\n# vi: set ft=python sts=4 ts=4 sw=4 et:\ndef configuration(parent_package='',top_path=None):\n    from numpy.distutils.misc_util import Configuration\n\n    config = Configuration('{pkg_name}', parent_package, top_path)\n\n    {sub_pks}\n\n    return config\n\nif __name__ == '__main__':\n    from numpy.distutils.core import setup\n    setup(**configuration(top_path='').todict())\n".format(pkg_name=package_dir.split('/')[-1], sub_pks='\n    '.join(["config.add_data_dir('%s')" % sub_pkg for sub_pkg in subpackages])))
            f.close()