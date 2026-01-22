from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
def validate_cache(self, cache_name=None):
    """
        Ensure the cached apidoc matches the one presented by the server.

        :param cache_name: The name of the apidoc on the server.
        """
    if cache_name is not None and cache_name != self.apidoc_cache_name:
        self.clean_cache()
        self.apidoc_cache_name = os.path.basename(os.path.normpath(cache_name))