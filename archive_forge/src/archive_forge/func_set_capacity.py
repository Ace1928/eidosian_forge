from boto.ec2.elb.listelement import ListElement
from boto.resultset import ResultSet
from boto.ec2.autoscale.launchconfig import LaunchConfiguration
from boto.ec2.autoscale.request import Request
from boto.ec2.autoscale.instance import Instance
from boto.ec2.autoscale.tag import Tag
def set_capacity(self, capacity):
    """
        Set the desired capacity for the group.
        """
    params = {'AutoScalingGroupName': self.name, 'DesiredCapacity': capacity}
    req = self.connection.get_object('SetDesiredCapacity', params, Request)
    self.connection.last_request = req
    return req