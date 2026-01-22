from boto.ec2.ec2object import TaggedEC2Object
from boto.ec2.zone import Zone
def get_permissions(self, dry_run=False):
    attrs = self.connection.get_snapshot_attribute(self.id, self.AttrName, dry_run=dry_run)
    return attrs.attrs