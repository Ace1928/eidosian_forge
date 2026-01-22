from __future__ import print_function
import os
import platform
import remove_pyreadline
import setuptools.command.easy_install as easy_install
import setuptools.package_index
import shutil
import sys
def remove_package(pkg):
    site_packages_dir, egg_name = os.path.split(pkg.location)
    easy_install_pth_filename = os.path.join(site_packages_dir, EASY_INSTALL_PTH_FILENAME)
    backup_filename = easy_install_pth_filename + BACKUP_SUFFIX
    shutil.copy2(easy_install_pth_filename, backup_filename)
    pth_file = easy_install.PthDistributions(easy_install_pth_filename)
    pth_file.remove(pkg)
    pth_file.save()
    if os.path.isdir(pkg.location):
        shutil.rmtree(pkg.location)
    else:
        os.unlink(pkg.location)