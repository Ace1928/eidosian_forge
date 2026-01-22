from __future__ import print_function
import os
import platform
import remove_pyreadline
import setuptools.command.easy_install as easy_install
import setuptools.package_index
import shutil
import sys
def find_package_consumers(name, deps_to_ignore=None):
    installed_packages = list(setuptools.package_index.AvailableDistributions())
    if deps_to_ignore is None:
        deps_to_ignore = []
    consumers = []
    for package_name in installed_packages:
        if name == package_name:
            continue
        package_info = setuptools.package_index.get_distribution(package_name)
        if package_name in deps_to_ignore:
            continue
        for req in package_info.requires():
            if req.project_name == name:
                consumers.append(package_name)
                break
    return consumers