import os
import boto.glacier
from boto.compat import json
from boto.connection import AWSAuthConnection
from boto.glacier.exceptions import UnexpectedHTTPResponseError
from boto.glacier.response import GlacierResponse
from boto.glacier.utils import ResettingFileSender
def create_vault(self, vault_name):
    """
        This operation creates a new vault with the specified name.
        The name of the vault must be unique within a region for an
        AWS account. You can create up to 1,000 vaults per account. If
        you need to create more vaults, contact Amazon Glacier.

        You must use the following guidelines when naming a vault.



        + Names can be between 1 and 255 characters long.
        + Allowed characters are a-z, A-Z, 0-9, '_' (underscore), '-'
          (hyphen), and '.' (period).



        This operation is idempotent.

        An AWS account has full permission to perform all operations
        (actions). However, AWS Identity and Access Management (IAM)
        users don't have any permissions by default. You must grant
        them explicit permission to perform specific actions. For more
        information, see `Access Control Using AWS Identity and Access
        Management (IAM)`_.

        For conceptual information and underlying REST API, go to
        `Creating a Vault in Amazon Glacier`_ and `Create Vault `_ in
        the Amazon Glacier Developer Guide .

        :type vault_name: string
        :param vault_name: The name of the vault.
        """
    uri = 'vaults/%s' % vault_name
    return self.make_request('PUT', uri, ok_responses=(201,), response_headers=[('Location', 'Location')])