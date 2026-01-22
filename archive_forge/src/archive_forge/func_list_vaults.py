import os
import boto.glacier
from boto.compat import json
from boto.connection import AWSAuthConnection
from boto.glacier.exceptions import UnexpectedHTTPResponseError
from boto.glacier.response import GlacierResponse
from boto.glacier.utils import ResettingFileSender
def list_vaults(self, limit=None, marker=None):
    """
        This operation lists all vaults owned by the calling user's
        account. The list returned in the response is ASCII-sorted by
        vault name.

        By default, this operation returns up to 1,000 items. If there
        are more vaults to list, the response `marker` field contains
        the vault Amazon Resource Name (ARN) at which to continue the
        list with a new List Vaults request; otherwise, the `marker`
        field is `null`. To return a list of vaults that begins at a
        specific vault, set the `marker` request parameter to the
        vault ARN you obtained from a previous List Vaults request.
        You can also limit the number of vaults returned in the
        response by specifying the `limit` parameter in the request.

        An AWS account has full permission to perform all operations
        (actions). However, AWS Identity and Access Management (IAM)
        users don't have any permissions by default. You must grant
        them explicit permission to perform specific actions. For more
        information, see `Access Control Using AWS Identity and Access
        Management (IAM)`_.

        For conceptual information and underlying REST API, go to
        `Retrieving Vault Metadata in Amazon Glacier`_ and `List
        Vaults `_ in the Amazon Glacier Developer Guide .

        :type marker: string
        :param marker: A string used for pagination. The marker specifies the
            vault ARN after which the listing of vaults should begin.

        :type limit: string
        :param limit: The maximum number of items returned in the response. If
            you don't specify a value, the List Vaults operation returns up to
            1,000 items.
        """
    params = {}
    if limit:
        params['limit'] = limit
    if marker:
        params['marker'] = marker
    return self.make_request('GET', 'vaults', params=params)