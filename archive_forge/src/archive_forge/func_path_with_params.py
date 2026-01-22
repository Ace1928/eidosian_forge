from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
def path_with_params(self, params=None):
    """
        Fill in the params into the path.

        :returns: The path with params.
        """
    result = self.path
    if params is not None:
        for param in self.params_in_path:
            if param in params:
                result = result.replace(':{}'.format(param), quote(str(params[param]), safe=''))
            else:
                raise KeyError("missing param '{}' in parameters".format(param))
    return result