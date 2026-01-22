import logging
import os
import sys
import warnings
from distutils import errors
from pbr import util
def pbr(dist, attr, value):
    """Implements the actual pbr setup() keyword.

    When used, this should be the only keyword in your setup() aside from
    `setup_requires`.

    If given as a string, the value of pbr is assumed to be the relative path
    to the setup.cfg file to use.  Otherwise, if it evaluates to true, it
    simply assumes that pbr should be used, and the default 'setup.cfg' is
    used.

    This works by reading the setup.cfg file, parsing out the supported
    metadata and command options, and using them to rebuild the
    `DistributionMetadata` object and set the newly added command options.

    The reason for doing things this way is that a custom `Distribution` class
    will not play nicely with setup_requires; however, this implementation may
    not work well with distributions that do use a `Distribution` subclass.
    """
    if hasattr(dist, '_pbr_initialized'):
        return
    dist._pbr_initialized = True
    if not value:
        return
    if isinstance(value, string_type):
        path = os.path.abspath(value)
    else:
        path = os.path.abspath('setup.cfg')
    if not os.path.exists(path):
        raise errors.DistutilsFileError('The setup.cfg file %s does not exist.' % path)
    try:
        attrs = util.cfg_to_args(path, dist.script_args)
    except Exception:
        e = sys.exc_info()[1]
        logging.exception('Error parsing')
        raise errors.DistutilsSetupError('Error parsing %s: %s: %s' % (path, e.__class__.__name__, e))
    _DISTUTILS_UNSUPPORTED_METADATA = ('long_description_content_type', 'project_urls', 'provides_extras')
    if attrs:
        for key, val in attrs.items():
            if hasattr(dist.metadata, 'set_' + key):
                getattr(dist.metadata, 'set_' + key)(val)
            elif hasattr(dist.metadata, key):
                setattr(dist.metadata, key, val)
            elif hasattr(dist, key):
                setattr(dist, key, val)
            elif key in _DISTUTILS_UNSUPPORTED_METADATA:
                setattr(dist.metadata, key, val)
            else:
                msg = 'Unknown distribution option: %s' % repr(key)
                warnings.warn(msg)
    try:
        super(dist.__class__, dist).finalize_options()
    except TypeError:
        dist.__class__.__bases__[-1].finalize_options(dist)
    if isinstance(dist.metadata.version, integer_types + (float,)):
        dist.metadata.version = str(dist.metadata.version)