import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.codedeploy import exceptions
def list_application_revisions(self, application_name, sort_by=None, sort_order=None, s_3_bucket=None, s_3_key_prefix=None, deployed=None, next_token=None):
    """
        Lists information about revisions for an application.

        :type application_name: string
        :param application_name: The name of an existing AWS CodeDeploy
            application within the AWS user account.

        :type sort_by: string
        :param sort_by: The column name to sort the list results by:

        + registerTime: Sort the list results by when the revisions were
              registered with AWS CodeDeploy.
        + firstUsedTime: Sort the list results by when the revisions were first
              used by in a deployment.
        + lastUsedTime: Sort the list results by when the revisions were last
              used in a deployment.


        If not specified or set to null, the results will be returned in an
            arbitrary order.

        :type sort_order: string
        :param sort_order: The order to sort the list results by:

        + ascending: Sort the list results in ascending order.
        + descending: Sort the list results in descending order.


        If not specified, the results will be sorted in ascending order.

        If set to null, the results will be sorted in an arbitrary order.

        :type s_3_bucket: string
        :param s_3_bucket: A specific Amazon S3 bucket name to limit the search
            for revisions.
        If set to null, then all of the user's buckets will be searched.

        :type s_3_key_prefix: string
        :param s_3_key_prefix: A specific key prefix for the set of Amazon S3
            objects to limit the search for revisions.

        :type deployed: string
        :param deployed:
        Whether to list revisions based on whether the revision is the target
            revision of an deployment group:


        + include: List revisions that are target revisions of a deployment
              group.
        + exclude: Do not list revisions that are target revisions of a
              deployment group.
        + ignore: List all revisions, regardless of whether they are target
              revisions of a deployment group.

        :type next_token: string
        :param next_token: An identifier that was returned from the previous
            list application revisions call, which can be used to return the
            next set of applications in the list.

        """
    params = {'applicationName': application_name}
    if sort_by is not None:
        params['sortBy'] = sort_by
    if sort_order is not None:
        params['sortOrder'] = sort_order
    if s_3_bucket is not None:
        params['s3Bucket'] = s_3_bucket
    if s_3_key_prefix is not None:
        params['s3KeyPrefix'] = s_3_key_prefix
    if deployed is not None:
        params['deployed'] = deployed
    if next_token is not None:
        params['nextToken'] = next_token
    return self.make_request(action='ListApplicationRevisions', body=json.dumps(params))