import sys
import json, os, sysconfig
def get_install_paths():
    if sys.version_info >= (3, 10):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    if sys.version_info >= (3, 10, 3):
        if 'deb_system' in sysconfig.get_scheme_names():
            scheme = 'deb_system'
    else:
        import distutils.command.install
        if 'deb_system' in distutils.command.install.INSTALL_SCHEMES:
            paths = get_distutils_paths(scheme='deb_system')
            install_paths = get_distutils_paths(scheme='deb_system', prefix='')
            return (paths, install_paths)
    paths = sysconfig.get_paths(scheme=scheme)
    empty_vars = {'base': '', 'platbase': '', 'installed_base': ''}
    install_paths = sysconfig.get_paths(scheme=scheme, vars=empty_vars)
    return (paths, install_paths)