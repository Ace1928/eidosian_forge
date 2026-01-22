import datetime
import numbers
import re
import sys
import os
import textwrap
from tornado.escape import _unicode, native_str
from tornado.log import define_logging_options
from tornado.util import basestring_type, exec_in
from typing import (
def parse_config_file(self, path: str, final: bool=True) -> None:
    """Parses and loads the config file at the given path.

        The config file contains Python code that will be executed (so
        it is **not safe** to use untrusted config files). Anything in
        the global namespace that matches a defined option will be
        used to set that option's value.

        Options may either be the specified type for the option or
        strings (in which case they will be parsed the same way as in
        `.parse_command_line`)

        Example (using the options defined in the top-level docs of
        this module)::

            port = 80
            mysql_host = 'mydb.example.com:3306'
            # Both lists and comma-separated strings are allowed for
            # multiple=True.
            memcache_hosts = ['cache1.example.com:11011',
                              'cache2.example.com:11011']
            memcache_hosts = 'cache1.example.com:11011,cache2.example.com:11011'

        If ``final`` is ``False``, parse callbacks will not be run.
        This is useful for applications that wish to combine configurations
        from multiple sources.

        .. note::

            `tornado.options` is primarily a command-line library.
            Config file support is provided for applications that wish
            to use it, but applications that prefer config files may
            wish to look at other libraries instead.

        .. versionchanged:: 4.1
           Config files are now always interpreted as utf-8 instead of
           the system default encoding.

        .. versionchanged:: 4.4
           The special variable ``__file__`` is available inside config
           files, specifying the absolute path to the config file itself.

        .. versionchanged:: 5.1
           Added the ability to set options via strings in config files.

        """
    config = {'__file__': os.path.abspath(path)}
    with open(path, 'rb') as f:
        exec_in(native_str(f.read()), config, config)
    for name in config:
        normalized = self._normalize_name(name)
        if normalized in self._options:
            option = self._options[normalized]
            if option.multiple:
                if not isinstance(config[name], (list, str)):
                    raise Error('Option %r is required to be a list of %s or a comma-separated string' % (option.name, option.type.__name__))
            if type(config[name]) == str and (option.type != str or option.multiple):
                option.parse(config[name])
            else:
                option.set(config[name])
    if final:
        self.run_parse_callbacks()