import sys
import json, os, sysconfig
def get_distutils_paths(scheme=None, prefix=None):
    import distutils.dist
    distribution = distutils.dist.Distribution()
    install_cmd = distribution.get_command_obj('install')
    if prefix is not None:
        install_cmd.prefix = prefix
    if scheme:
        install_cmd.select_scheme(scheme)
    install_cmd.finalize_options()
    return {'data': install_cmd.install_data, 'include': os.path.dirname(install_cmd.install_headers), 'platlib': install_cmd.install_platlib, 'purelib': install_cmd.install_purelib, 'scripts': install_cmd.install_scripts}