from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
def validate_path_params(self, operation_name, params):
    """
        Validate params for the get requests. Use this method for validating the path part of the url.
           :param operation_name: string
                               The value must be non empty string.
                               The operation name is used to get a params specification
           :param params: dict
                        should be in the format that the specification(from operation) expects

                 Ex.
                 {
                     'objId': "string_value",
                     'p_integer': 1,
                     'p_boolean': True,
                     'p_number': 2.3
                 }
        :rtype:(Boolean, msg)
        :return:
            (True, None) - if params valid
            Invalid:
            (False, {
                'required': [ #list of the fields that are required but are not present in the params
                    'field_name'
                ],
                'invalid_type':[ #list of the fields with invalid data and expected type of the params
                         {
                           'path': 'objId', #field name
                           'expected_type': 'string',#expected type. Ex. 'string', 'integer', 'boolean', 'number'
                           'actually_value': 1 # the value that user passed
                         }
                ]
            })
        :raises IllegalArgumentException
            'The operation_name parameter must be a non-empty string' if operation_name is not valid
            'The params parameter must be a dict' if params neither dict or None
            '{operation_name} operation does not support' if the spec does not contain the operation
        """
    return self._validate_url_params(operation_name, params, resource=OperationParams.PATH)