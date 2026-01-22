import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.redshift import exceptions
def modify_cluster_parameter_group(self, parameter_group_name, parameters):
    """
        Modifies the parameters of a parameter group.

        For more information about managing parameter groups, go to
        `Amazon Redshift Parameter Groups`_ in the Amazon Redshift
        Management Guide .

        :type parameter_group_name: string
        :param parameter_group_name: The name of the parameter group to be
            modified.

        :type parameters: list
        :param parameters: An array of parameters to be modified. A maximum of
            20 parameters can be modified in a single request.
        For each parameter to be modified, you must supply at least the
            parameter name and parameter value; other name-value pairs of the
            parameter are optional.

        For the workload management (WLM) configuration, you must supply all
            the name-value pairs in the wlm_json_configuration parameter.

        """
    params = {'ParameterGroupName': parameter_group_name}
    self.build_complex_list_params(params, parameters, 'Parameters.member', ('ParameterName', 'ParameterValue', 'Description', 'Source', 'DataType', 'AllowedValues', 'IsModifiable', 'MinimumEngineVersion'))
    return self._make_request(action='ModifyClusterParameterGroup', verb='POST', path='/', params=params)