import os
import textwrap
from optparse import Values
from typing import Any, List
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.exceptions import CommandError, PipError
from pip._internal.utils import filesystem
from pip._internal.utils.logging import getLogger
def get_cache_info(self, options: Values, args: List[Any]) -> None:
    if args:
        raise CommandError('Too many arguments')
    num_http_files = len(self._find_http_files(options))
    num_packages = len(self._find_wheels(options, '*'))
    http_cache_location = self._cache_dir(options, 'http-v2')
    old_http_cache_location = self._cache_dir(options, 'http')
    wheels_cache_location = self._cache_dir(options, 'wheels')
    http_cache_size = filesystem.format_size(filesystem.directory_size(http_cache_location) + filesystem.directory_size(old_http_cache_location))
    wheels_cache_size = filesystem.format_directory_size(wheels_cache_location)
    message = textwrap.dedent('\n                    Package index page cache location (pip v23.3+): {http_cache_location}\n                    Package index page cache location (older pips): {old_http_cache_location}\n                    Package index page cache size: {http_cache_size}\n                    Number of HTTP files: {num_http_files}\n                    Locally built wheels location: {wheels_cache_location}\n                    Locally built wheels size: {wheels_cache_size}\n                    Number of locally built wheels: {package_count}\n                ').format(http_cache_location=http_cache_location, old_http_cache_location=old_http_cache_location, http_cache_size=http_cache_size, num_http_files=num_http_files, wheels_cache_location=wheels_cache_location, package_count=num_packages, wheels_cache_size=wheels_cache_size).strip()
    logger.info(message)