import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.configservice import exceptions
def stop_configuration_recorder(self, configuration_recorder_name):
    """
        Stops recording configurations of all the resources associated
        with the account.

        :type configuration_recorder_name: string
        :param configuration_recorder_name: The name of the recorder object
            that records each configuration change made to the resources.

        """
    params = {'ConfigurationRecorderName': configuration_recorder_name}
    return self.make_request(action='StopConfigurationRecorder', body=json.dumps(params))