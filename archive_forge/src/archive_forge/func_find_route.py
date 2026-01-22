from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
def find_route(self, params=None):
    """
        Find the best matching route for a given set of params.

        :param params: Params that should be submitted to the API.

        :returns: The best route.
        """
    param_keys = set(self.filter_empty_params(params).keys())
    sorted_routes = sorted(self.routes, key=lambda route: [-1 * len(route.params_in_path), route.path])
    for route in sorted_routes:
        if set(route.params_in_path) <= param_keys:
            return route
    return sorted_routes[-1]