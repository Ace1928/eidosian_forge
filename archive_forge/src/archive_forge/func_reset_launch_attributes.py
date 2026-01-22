from boto.ec2.ec2object import EC2Object, TaggedEC2Object
from boto.ec2.blockdevicemapping import BlockDeviceMapping
def reset_launch_attributes(self, dry_run=False):
    return self.connection.reset_image_attribute(self.id, 'launchPermission', dry_run=dry_run)