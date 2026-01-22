from (runtime, environment) to arbitrary data. Its main feature is that it
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from six.moves import map  # pylint:disable=redefined-builtin
Return the associated value for the given runtime/environment.

    Args:
      runtime: str, the runtime to get a stager for
      env: env, the environment to get a stager for

    Returns:
      object, the matching entry, or override if one was specified. If no
        match is found, will return default if specified or None otherwise.
    