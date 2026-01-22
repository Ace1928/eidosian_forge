import boto
import boto.jsonresponse
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def describe_index_fields(self, domain_name, field_names=None):
    """
        Describes index fields in the search domain, optionally
        limited to a single ``IndexField``.

        :type domain_name: string
        :param domain_name: A string that represents the name of a
            domain. Domain names must be unique across the domains
            owned by an account within an AWS region. Domain names
            must start with a letter or number and can contain the
            following characters: a-z (lowercase), 0-9, and -
            (hyphen). Uppercase letters and underscores are not
            allowed.

        :type field_names: list
        :param field_names: Limits the response to the specified fields.

        :raises: BaseException, InternalException, ResourceNotFoundException
        """
    doc_path = ('describe_index_fields_response', 'describe_index_fields_result', 'index_fields')
    params = {'DomainName': domain_name}
    if field_names:
        for i, field_name in enumerate(field_names, 1):
            params['FieldNames.member.%d' % i] = field_name
    return self.get_response(doc_path, 'DescribeIndexFields', params, verb='POST', list_marker='IndexFields')