import boto
import boto.utils
from boto.compat import StringIO
from boto.mashups.iobject import IObject
from boto.pyami.config import Config, BotoConfigPath
from boto.mashups.interactive import interactive_shell
from boto.sdb.db.model import Model
from boto.sdb.db.property import StringProperty
import os
def detach_volume(self, volume):
    """
        Detach an EBS volume from this server

        :param volume: EBS Volume to detach
        :type volume: boto.ec2.volume.Volume
        """
    if hasattr(volume, 'id'):
        volume_id = volume.id
    else:
        volume_id = volume
    return self.ec2.detach_volume(volume_id=volume_id, instance_id=self.instance_id)