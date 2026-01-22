import boto
from boto.gs.bucket import Bucket
from boto.s3.connection import S3Connection
from boto.s3.connection import SubdomainCallingFormat
from boto.s3.connection import check_lowercase_bucketname
from boto.compat import six
from boto.utils import get_utf8able_str
def create_bucket(self, bucket_name, headers=None, location=Location.DEFAULT, policy=None, storage_class='STANDARD'):
    """
        Creates a new bucket. By default it's located in the USA. You can
        pass Location.EU to create bucket in the EU. You can also pass
        a LocationConstraint for where the bucket should be located, and
        a StorageClass describing how the data should be stored.

        :type bucket_name: string
        :param bucket_name: The name of the new bucket.

        :type headers: dict
        :param headers: Additional headers to pass along with the request to GCS.

        :type location: :class:`boto.gs.connection.Location`
        :param location: The location of the new bucket.

        :type policy: :class:`boto.gs.acl.CannedACLStrings`
        :param policy: A canned ACL policy that will be applied to the new key
                       in GCS.

        :type storage_class: string
        :param storage_class: Either 'STANDARD' or 'DURABLE_REDUCED_AVAILABILITY'.

        """
    check_lowercase_bucketname(bucket_name)
    if policy:
        if headers:
            headers[self.provider.acl_header] = policy
        else:
            headers = {self.provider.acl_header: policy}
    if not location:
        location = Location.DEFAULT
    location_elem = '<LocationConstraint>%s</LocationConstraint>' % location
    if storage_class:
        storage_class_elem = '<StorageClass>%s</StorageClass>' % storage_class
    else:
        storage_class_elem = ''
    data = '<CreateBucketConfiguration>%s%s</CreateBucketConfiguration>' % (location_elem, storage_class_elem)
    response = self.make_request('PUT', get_utf8able_str(bucket_name), headers=headers, data=get_utf8able_str(data))
    body = response.read()
    if response.status == 409:
        raise self.provider.storage_create_error(response.status, response.reason, body)
    if response.status == 200:
        return self.bucket_class(self, bucket_name)
    else:
        raise self.provider.storage_response_error(response.status, response.reason, body)