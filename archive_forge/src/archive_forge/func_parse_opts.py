import abc
import argparse
import os
from zunclient.common.apiclient import exceptions
def parse_opts(self, args):
    """Parse the actual auth-system options if any.

        This method is expected to populate the attribute `self.opts` with a
        dict containing the options and values needed to make authentication.
        """
    self.opts.update(dict((self.get_opt(opt_name, args) for opt_name in self.opt_names)))