from boto.ec2.elb.listelement import ListElement
from boto.resultset import ResultSet
from boto.ec2.autoscale.launchconfig import LaunchConfiguration
from boto.ec2.autoscale.request import Request
from boto.ec2.autoscale.instance import Instance
from boto.ec2.autoscale.tag import Tag
def shutdown_instances(self):
    """
        Convenience method which shuts down all instances associated with
        this group.
        """
    self.min_size = 0
    self.max_size = 0
    self.desired_capacity = 0
    self.update()