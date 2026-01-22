import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.rds2 import exceptions
from boto.compat import json
def modify_db_parameter_group(self, db_parameter_group_name, parameters):
    """
        Modifies the parameters of a DB parameter group. To modify
        more than one parameter, submit a list of the following:
        `ParameterName`, `ParameterValue`, and `ApplyMethod`. A
        maximum of 20 parameters can be modified in a single request.

        The `apply-immediate` method can be used only for dynamic
        parameters; the `pending-reboot` method can be used with MySQL
        and Oracle DB instances for either dynamic or static
        parameters. For Microsoft SQL Server DB instances, the
        `pending-reboot` method can be used only for static
        parameters.

        :type db_parameter_group_name: string
        :param db_parameter_group_name:
        The name of the DB parameter group.

        Constraints:


        + Must be the name of an existing DB parameter group
        + Must be 1 to 255 alphanumeric characters
        + First character must be a letter
        + Cannot end with a hyphen or contain two consecutive hyphens

        :type parameters: list
        :param parameters:
        An array of parameter names, values, and the apply method for the
            parameter update. At least one parameter name, value, and apply
            method must be supplied; subsequent arguments are optional. A
            maximum of 20 parameters may be modified in a single request.

        Valid Values (for the application method): `immediate | pending-reboot`

        You can use the immediate value with dynamic parameters only. You can
            use the pending-reboot value for both dynamic and static
            parameters, and changes are applied when DB instance reboots.

        """
    params = {'DBParameterGroupName': db_parameter_group_name}
    self.build_complex_list_params(params, parameters, 'Parameters.member', ('ParameterName', 'ParameterValue', 'Description', 'Source', 'ApplyType', 'DataType', 'AllowedValues', 'IsModifiable', 'MinimumEngineVersion', 'ApplyMethod'))
    return self._make_request(action='ModifyDBParameterGroup', verb='POST', path='/', params=params)