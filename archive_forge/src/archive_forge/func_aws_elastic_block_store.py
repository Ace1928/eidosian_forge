from pprint import pformat
from six import iteritems
import re
@aws_elastic_block_store.setter
def aws_elastic_block_store(self, aws_elastic_block_store):
    """
        Sets the aws_elastic_block_store of this V1PersistentVolumeSpec.
        AWSElasticBlockStore represents an AWS Disk resource that is attached to
        a kubelet's host machine and then exposed to the pod. More info:
        https://kubernetes.io/docs/concepts/storage/volumes#awselasticblockstore

        :param aws_elastic_block_store: The aws_elastic_block_store of this
        V1PersistentVolumeSpec.
        :type: V1AWSElasticBlockStoreVolumeSource
        """
    self._aws_elastic_block_store = aws_elastic_block_store