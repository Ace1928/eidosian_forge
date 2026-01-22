from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
@property
def terminal_job_states(self):
    return [self.messages.JobStatus.StateValueValuesEnum.CANCELLED, self.messages.JobStatus.StateValueValuesEnum.DONE, self.messages.JobStatus.StateValueValuesEnum.ERROR]