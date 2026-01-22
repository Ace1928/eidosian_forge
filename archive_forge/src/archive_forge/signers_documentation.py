import base64
import datetime
import json
import weakref
import botocore
import botocore.auth
from botocore.awsrequest import create_request_object, prepare_request_dict
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import ArnParser, datetime2timestamp
from botocore.utils import fix_s3_host  # noqa
Generates the url and the form fields used for a presigned s3 post

        :type request_dict: dict
        :param request_dict: The prepared request dictionary returned by
            ``botocore.awsrequest.prepare_request_dict()``

        :type fields: dict
        :param fields: A dictionary of prefilled form fields to build on top
            of.

        :type conditions: list
        :param conditions: A list of conditions to include in the policy. Each
            element can be either a list or a structure. For example:
            [
             {"acl": "public-read"},
             {"bucket": "mybucket"},
             ["starts-with", "$key", "mykey"]
            ]

        :type expires_in: int
        :param expires_in: The number of seconds the presigned post is valid
            for.

        :type region_name: string
        :param region_name: The region name to sign the presigned post to.

        :rtype: dict
        :returns: A dictionary with two elements: ``url`` and ``fields``.
            Url is the url to post to. Fields is a dictionary filled with
            the form fields and respective values to use when submitting the
            post. For example:

            {'url': 'https://mybucket.s3.amazonaws.com
             'fields': {'acl': 'public-read',
                        'key': 'mykey',
                        'signature': 'mysignature',
                        'policy': 'mybase64 encoded policy'}
            }
        