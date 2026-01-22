from boto.ec2.ec2object import EC2Object, TaggedEC2Object
from boto.ec2.blockdevicemapping import BlockDeviceMapping
def get_kernel(self, dry_run=False):
    img_attrs = self.connection.get_image_attribute(self.id, 'kernel', dry_run=dry_run)
    return img_attrs.kernel