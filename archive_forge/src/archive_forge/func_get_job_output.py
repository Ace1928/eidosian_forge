import os
import boto.glacier
from boto.compat import json
from boto.connection import AWSAuthConnection
from boto.glacier.exceptions import UnexpectedHTTPResponseError
from boto.glacier.response import GlacierResponse
from boto.glacier.utils import ResettingFileSender
def get_job_output(self, vault_name, job_id, byte_range=None):
    """
        This operation downloads the output of the job you initiated
        using InitiateJob. Depending on the job type you specified
        when you initiated the job, the output will be either the
        content of an archive or a vault inventory.

        A job ID will not expire for at least 24 hours after Amazon
        Glacier completes the job. That is, you can download the job
        output within the 24 hours period after Amazon Glacier
        completes the job.

        If the job output is large, then you can use the `Range`
        request header to retrieve a portion of the output. This
        allows you to download the entire output in smaller chunks of
        bytes. For example, suppose you have 1 GB of job output you
        want to download and you decide to download 128 MB chunks of
        data at a time, which is a total of eight Get Job Output
        requests. You use the following process to download the job
        output:


        #. Download a 128 MB chunk of output by specifying the
           appropriate byte range using the `Range` header.
        #. Along with the data, the response includes a checksum of
           the payload. You compute the checksum of the payload on the
           client and compare it with the checksum you received in the
           response to ensure you received all the expected data.
        #. Repeat steps 1 and 2 for all the eight 128 MB chunks of
           output data, each time specifying the appropriate byte range.
        #. After downloading all the parts of the job output, you have
           a list of eight checksum values. Compute the tree hash of
           these values to find the checksum of the entire output. Using
           the Describe Job API, obtain job information of the job that
           provided you the output. The response includes the checksum of
           the entire archive stored in Amazon Glacier. You compare this
           value with the checksum you computed to ensure you have
           downloaded the entire archive content with no errors.


        An AWS account has full permission to perform all operations
        (actions). However, AWS Identity and Access Management (IAM)
        users don't have any permissions by default. You must grant
        them explicit permission to perform specific actions. For more
        information, see `Access Control Using AWS Identity and Access
        Management (IAM)`_.

        For conceptual information and the underlying REST API, go to
        `Downloading a Vault Inventory`_, `Downloading an Archive`_,
        and `Get Job Output `_

        :type account_id: string
        :param account_id: The `AccountId` is the AWS Account ID. You can
            specify either the AWS Account ID or optionally a '-', in which
            case Amazon Glacier uses the AWS Account ID associated with the
            credentials used to sign the request. If you specify your Account
            ID, do not include hyphens in it.

        :type vault_name: string
        :param vault_name: The name of the vault.

        :type job_id: string
        :param job_id: The job ID whose data is downloaded.

        :type byte_range: string
        :param byte_range: The range of bytes to retrieve from the output. For
            example, if you want to download the first 1,048,576 bytes, specify
            "Range: bytes=0-1048575". By default, this operation downloads the
            entire output.
        """
    response_headers = [('x-amz-sha256-tree-hash', u'TreeHash'), ('Content-Range', u'ContentRange'), ('Content-Type', u'ContentType')]
    headers = None
    if byte_range:
        headers = {'Range': 'bytes=%d-%d' % byte_range}
    uri = 'vaults/%s/jobs/%s/output' % (vault_name, job_id)
    response = self.make_request('GET', uri, headers=headers, ok_responses=(200, 206), response_headers=response_headers)
    return response