import abc
import argparse
import os
from zunclient.common.apiclient import exceptions
def sufficient_options(self):
    """Check if all required options are present.

        :raises: AuthPluginOptionsMissing
        """
    missing = [opt for opt in self.opt_names if not self.opts.get(opt)]
    if missing:
        raise exceptions.AuthPluginOptionsMissing(missing)