import boto
import boto.jsonresponse
from boto.compat import json
from boto.regioninfo import RegionInfo
from boto.connection import AWSQueryConnection
def swap_environment_cnames(self, source_environment_id=None, source_environment_name=None, destination_environment_id=None, destination_environment_name=None):
    """Swaps the CNAMEs of two environments.

        :type source_environment_id: string
        :param source_environment_id: The ID of the source environment.
            Condition: You must specify at least the SourceEnvironmentID or the
            SourceEnvironmentName. You may also specify both. If you specify
            the SourceEnvironmentId, you must specify the
            DestinationEnvironmentId.

        :type source_environment_name: string
        :param source_environment_name: The name of the source environment.
            Condition: You must specify at least the SourceEnvironmentID or the
            SourceEnvironmentName. You may also specify both. If you specify
            the SourceEnvironmentName, you must specify the
            DestinationEnvironmentName.

        :type destination_environment_id: string
        :param destination_environment_id: The ID of the destination
            environment.  Condition: You must specify at least the
            DestinationEnvironmentID or the DestinationEnvironmentName. You may
            also specify both. You must specify the SourceEnvironmentId with
            the DestinationEnvironmentId.

        :type destination_environment_name: string
        :param destination_environment_name: The name of the destination
            environment.  Condition: You must specify at least the
            DestinationEnvironmentID or the DestinationEnvironmentName. You may
            also specify both. You must specify the SourceEnvironmentName with
            the DestinationEnvironmentName.
        """
    params = {}
    if source_environment_id:
        params['SourceEnvironmentId'] = source_environment_id
    if source_environment_name:
        params['SourceEnvironmentName'] = source_environment_name
    if destination_environment_id:
        params['DestinationEnvironmentId'] = destination_environment_id
    if destination_environment_name:
        params['DestinationEnvironmentName'] = destination_environment_name
    return self._get_response('SwapEnvironmentCNAMEs', params)