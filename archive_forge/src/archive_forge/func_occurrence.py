from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.artifacts.vulnerabilities import GetLatestScan
from googlecloudsdk.api_lib.artifacts.vulnerabilities import GetVulnerabilities
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.artifacts import flags
from googlecloudsdk.command_lib.artifacts import format_util
@property
def occurrence(self):
    return self._occurrence