import sys
import os
import boto
import optparse
import copy
import boto.exception
import boto.roboto.awsqueryservice
import bdb
import traceback
def cli_formatter(self, data):
    """
        This method is responsible for formatting the output for the
        command line interface.  The default behavior is to call the
        generic CLI formatter which attempts to print something
        reasonable.  If you want specific formatting, you should
        override this method and do your own thing.

        :type data: dict
        :param data: The data returned by AWS.
        """
    if data:
        self._generic_cli_formatter(self.Response, data)