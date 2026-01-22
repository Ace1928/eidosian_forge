import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cloudsearch2 import exceptions
def build_complex_param(self, params, label, value):
    """Serialize a structure.

        For example::

            param_type = 'structure'
            label = 'IndexField'
            value = {'IndexFieldName': 'a', 'IntOptions': {'DefaultValue': 5}}

        would result in the params dict being updated with these params::

            IndexField.IndexFieldName = a
            IndexField.IntOptions.DefaultValue = 5

        :type params: dict
        :param params: The params dict.  The complex list params
            will be added to this dict.

        :type label: str
        :param label: String label for param key

        :type value: any
        :param value: The value to serialize
        """
    for k, v in value.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                self.build_complex_param(params, label + '.' + k, v)
        elif isinstance(v, bool):
            params['%s.%s' % (label, k)] = v and 'true' or 'false'
        else:
            params['%s.%s' % (label, k)] = v