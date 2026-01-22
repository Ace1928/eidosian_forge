import logging  # noqa
from collections import defaultdict
import io
import os
import re
import shlex
import sys
import traceback
import distutils.ccompiler
from distutils import errors
from distutils import log
import pkg_resources
from setuptools import dist as st_dist
from setuptools import extension
from pbr import extra_files
import pbr.hooks
def cfg_to_args(path='setup.cfg', script_args=()):
    """Distutils2 to distutils1 compatibility util.

    This method uses an existing setup.cfg to generate a dictionary of
    keywords that can be used by distutils.core.setup(kwargs**).

    :param path:
        The setup.cfg path.
    :param script_args:
        List of commands setup.py was called with.
    :raises DistutilsFileError:
        When the setup.cfg file is not found.
    """
    if sys.version_info >= (3, 0):
        parser = configparser.ConfigParser()
    else:
        parser = configparser.SafeConfigParser()
    if not os.path.exists(path):
        raise errors.DistutilsFileError("file '%s' does not exist" % os.path.abspath(path))
    try:
        parser.read(path, encoding='utf-8')
    except TypeError:
        parser.read(path)
    config = {}
    for section in parser.sections():
        config[section] = dict()
        for k, value in parser.items(section):
            config[section][k.replace('-', '_')] = value
    setup_hooks = has_get_option(config, 'global', 'setup_hooks')
    package_dir = has_get_option(config, 'files', 'packages_root')
    if package_dir:
        package_dir = os.path.abspath(package_dir)
        sys.path.insert(0, package_dir)
    try:
        if setup_hooks:
            setup_hooks = [hook for hook in split_multiline(setup_hooks) if hook != 'pbr.hooks.setup_hook']
            for hook in setup_hooks:
                hook_fn = resolve_name(hook)
                try:
                    hook_fn(config)
                except SystemExit:
                    log.error('setup hook %s terminated the installation')
                except Exception:
                    e = sys.exc_info()[1]
                    log.error('setup hook %s raised exception: %s\n' % (hook, e))
                    log.error(traceback.format_exc())
                    sys.exit(1)
        pbr.hooks.setup_hook(config)
        kwargs = setup_cfg_to_setup_kwargs(config, script_args)
        kwargs['include_package_data'] = True
        kwargs['zip_safe'] = False
        register_custom_compilers(config)
        ext_modules = get_extension_modules(config)
        if ext_modules:
            kwargs['ext_modules'] = ext_modules
        entry_points = get_entry_points(config)
        if entry_points:
            kwargs['entry_points'] = entry_points
        files_extra_files = has_get_option(config, 'files', 'extra_files')
        if files_extra_files:
            extra_files.set_extra_files(split_multiline(files_extra_files))
    finally:
        if package_dir:
            sys.path.pop(0)
    return kwargs