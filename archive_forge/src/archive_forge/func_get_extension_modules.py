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
def get_extension_modules(config):
    """Handle extension modules"""
    EXTENSION_FIELDS = ('sources', 'include_dirs', 'define_macros', 'undef_macros', 'library_dirs', 'libraries', 'runtime_library_dirs', 'extra_objects', 'extra_compile_args', 'extra_link_args', 'export_symbols', 'swig_opts', 'depends')
    ext_modules = []
    for section in config:
        if ':' in section:
            labels = section.split(':', 1)
        else:
            labels = section.split('=', 1)
        labels = [label.strip() for label in labels]
        if len(labels) == 2 and labels[0] == 'extension':
            ext_args = {}
            for field in EXTENSION_FIELDS:
                value = has_get_option(config, section, field)
                if not value:
                    continue
                value = split_multiline(value)
                if field == 'define_macros':
                    macros = []
                    for macro in value:
                        macro = macro.split('=', 1)
                        if len(macro) == 1:
                            macro = (macro[0].strip(), None)
                        else:
                            macro = (macro[0].strip(), macro[1].strip())
                        macros.append(macro)
                    value = macros
                ext_args[field] = value
            if ext_args:
                if 'name' not in ext_args:
                    ext_args['name'] = labels[1]
                ext_modules.append(extension.Extension(ext_args.pop('name'), **ext_args))
    return ext_modules