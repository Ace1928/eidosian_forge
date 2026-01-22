from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.command_lib.storage import errors
import six
def get_urls_iterable(normal_urls_argument, should_read_paths_from_stdin, allow_empty=False):
    """Helps command decide between normal URL args and a StdinIterator."""
    if not (normal_urls_argument or should_read_paths_from_stdin or allow_empty):
        raise errors.InvalidUrlError('Must have URL arguments if not reading paths from stdin.')
    if normal_urls_argument and should_read_paths_from_stdin:
        raise errors.InvalidUrlError('Cannot have both read from stdin flag and normal URL arguments.')
    if should_read_paths_from_stdin:
        return StdinIterator()
    return normal_urls_argument