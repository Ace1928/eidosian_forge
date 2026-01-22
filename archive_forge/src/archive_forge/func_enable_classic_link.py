from boto.ec2.ec2object import TaggedEC2Object
def enable_classic_link(self, dry_run=False):
    """
        Enables a VPC for ClassicLink. You can then link EC2-Classic instances
        to your ClassicLink-enabled VPC to allow communication over private IP
        addresses. You cannot enable your VPC for ClassicLink if any of your
        VPC's route tables have existing routes for address ranges within the
        10.0.0.0/8 IP address range, excluding local routes for VPCs in the
        10.0.0.0/16 and 10.1.0.0/16 IP address ranges.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: bool
        :return: True if successful
        """
    return self.connection.enable_vpc_classic_link(self.id, dry_run=dry_run)