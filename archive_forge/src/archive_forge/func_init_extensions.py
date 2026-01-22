import glob
from itertools import chain
import os
import sys
from traitlets.config.application import boolean_flag
from traitlets.config.configurable import Configurable
from traitlets.config.loader import Config
from IPython.core.application import SYSTEM_CONFIG_DIRS, ENV_CONFIG_DIRS
from IPython.core import pylabtools
from IPython.utils.contexts import preserve_keys
from IPython.utils.path import filefind
from traitlets import (
from IPython.terminal import pt_inputhooks
def init_extensions(self):
    """Load all IPython extensions in IPythonApp.extensions.

        This uses the :meth:`ExtensionManager.load_extensions` to load all
        the extensions listed in ``self.extensions``.
        """
    try:
        self.log.debug('Loading IPython extensions...')
        extensions = self.default_extensions + self.extensions + self.extra_extensions
        for ext in extensions:
            try:
                self.log.info('Loading IPython extension: %s', ext)
                self.shell.extension_manager.load_extension(ext)
            except:
                if self.reraise_ipython_extension_failures:
                    raise
                msg = 'Error in loading extension: {ext}\nCheck your config files in {location}'.format(ext=ext, location=self.profile_dir.location)
                self.log.warning(msg, exc_info=True)
    except:
        if self.reraise_ipython_extension_failures:
            raise
        self.log.warning('Unknown error in loading extensions:', exc_info=True)