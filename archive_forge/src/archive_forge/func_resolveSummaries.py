from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.container.images import container_data_util
from googlecloudsdk.api_lib.containeranalysis import requests
import six
def resolveSummaries(self):
    self.package_vulnerability_summary.resolve()
    self.image_basis_summary.resolve()
    self.build_details_summary.resolve()
    self.deployment_summary.resolve()
    self.discovery_summary.resolve()