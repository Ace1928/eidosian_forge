import boto
import boto.jsonresponse
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
def describe_rank_expressions(self, domain_name, rank_names=None):
    """
        Describes RankExpressions in the search domain, optionally
        limited to a single expression.

        :type domain_name: string
        :param domain_name: A string that represents the name of a
            domain. Domain names must be unique across the domains
            owned by an account within an AWS region. Domain names
            must start with a letter or number and can contain the
            following characters: a-z (lowercase), 0-9, and -
            (hyphen). Uppercase letters and underscores are not
            allowed.

        :type rank_names: list
        :param rank_names: Limit response to the specified rank names.

        :raises: BaseException, InternalException, ResourceNotFoundException
        """
    doc_path = ('describe_rank_expressions_response', 'describe_rank_expressions_result', 'rank_expressions')
    params = {'DomainName': domain_name}
    if rank_names:
        for i, rank_name in enumerate(rank_names, 1):
            params['RankNames.member.%d' % i] = rank_name
    return self.get_response(doc_path, 'DescribeRankExpressions', params, verb='POST', list_marker='RankExpressions')