from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import time
from google.api_core import bidi
from google.rpc import error_details_pb2
from googlecloudsdk.api_lib.util import api_enablement
from googlecloudsdk.calliope import base
from googlecloudsdk.core import config
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport as core_transport
from googlecloudsdk.core.credentials import transport
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import http_proxy_types
import grpc
from six.moves import urllib
import socks
def log_response(self, response, time_taken):
    """Logs information about the request.

    Args:
        response: A grpc.Call/grpc.Future instance representing a service
            response.
        time_taken: time, in seconds, it took for the RPC to complete.
    """
    log.status.Print('---- response start ----')
    log.status.Print('code: {}'.format(response.code()))
    log.status.Print('-- headers start --')
    log.status.Print('details: {}'.format(response.details()))
    log.status.Print('-- initial metadata --')
    self.log_metadata(response.initial_metadata())
    log.status.Print('-- trailing metadata --')
    self.log_metadata(response.trailing_metadata())
    log.status.Print('-- headers end --')
    log.status.Print('-- body start --')
    log.status.Print('{}'.format(response.result()))
    log.status.Print('-- body end --')
    log.status.Print('total round trip time (request+response): {0:.3f} secs'.format(time_taken))
    log.status.Print('---- response end ----')
    log.status.Print('----------------------')