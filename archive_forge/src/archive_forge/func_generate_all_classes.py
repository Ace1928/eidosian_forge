import xml.dom.minidom
import subprocess
import os
from shutil import rmtree
import keyword
from ..base import (CommandLine, CommandLineInputSpec, SEMLikeCommandLine, TraitedSpec,
import os\n\n\n"""
def generate_all_classes(modules_list=[], launcher=[], redirect_x=False, mipav_hacks=False):
    """modules_list contains all the SEM compliant tools that should have wrappers created for them.
    launcher containtains the command line prefix wrapper arguments needed to prepare
    a proper environment for each of the modules.
    """
    all_code = {}
    for module in modules_list:
        print('=' * 80)
        print('Generating Definition for module {0}'.format(module))
        print('^' * 80)
        package, code, module = generate_class(module, launcher, redirect_x=redirect_x, mipav_hacks=mipav_hacks)
        cur_package = all_code
        module_name = package.strip().split(' ')[0].split('.')[-1]
        for package in package.strip().split(' ')[0].split('.')[:-1]:
            if package not in cur_package:
                cur_package[package] = {}
            cur_package = cur_package[package]
        if module_name not in cur_package:
            cur_package[module_name] = {}
        cur_package[module_name][module] = code
    if os.path.exists('__init__.py'):
        os.unlink('__init__.py')
    crawl_code_struct(all_code, os.getcwd())