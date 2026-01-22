from boto.ec2.ec2object import EC2Object
def disassociate(self, dry_run=False):
    """
        Disassociate this Elastic IP address from a currently running instance.
        :see: :meth:`boto.ec2.connection.EC2Connection.disassociate_address`
        """
    if self.association_id:
        return self.connection.disassociate_address(association_id=self.association_id, dry_run=dry_run)
    else:
        return self.connection.disassociate_address(public_ip=self.public_ip, dry_run=dry_run)