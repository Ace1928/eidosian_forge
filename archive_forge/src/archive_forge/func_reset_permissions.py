from boto.ec2.ec2object import TaggedEC2Object
from boto.ec2.zone import Zone
def reset_permissions(self, dry_run=False):
    return self.connection.reset_snapshot_attribute(self.id, self.AttrName, dry_run=dry_run)