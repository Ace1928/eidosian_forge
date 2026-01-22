import logging
from warnings import warn
import os
import sys
from .misc import str2bool
def logdebug_dict_differences(self, dold, dnew, prefix=''):
    """Helper to log what actually changed from old to new values of
        dictionaries.

        typical use -- log difference for hashed_inputs
        """
    from .misc import dict_diff
    self._logger.warning('logdebug_dict_differences has been deprecated, please use nipype.utils.misc.dict_diff.')
    self._logger.debug(dict_diff(dold, dnew))