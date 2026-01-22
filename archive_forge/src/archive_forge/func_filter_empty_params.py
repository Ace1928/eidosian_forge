from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
@staticmethod
def filter_empty_params(params=None):
    """
        Filter out any params that have no value.

        :param params: The params to filter.

        :returns: The filtered params.
        """
    result = {}
    if params is not None:
        if isinstance(params, dict):
            result = {k: v for k, v in params.items() if v is not None}
        else:
            raise InvalidArgumentTypesError
    return result