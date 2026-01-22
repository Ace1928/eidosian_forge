from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import encoding
from cloudsdk.google.protobuf import timestamp_pb2
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.command_lib.compute import ssh_troubleshooter_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console.console_io import OperationCancelledError
Create a MonitoringProjectsTimeSeriesListRequest.

    Args:
      metrics_type: str, https://cloud.google.com/monitoring/api/metrics

    Returns:
      MonitoringProjectsTimeSeriesListRequest, input message for
      ProjectsTimeSeriesService List method.
    