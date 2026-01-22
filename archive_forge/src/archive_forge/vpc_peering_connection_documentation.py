from boto.ec2.ec2object import TaggedEC2Object

        Represents a VPC peering connection.

        :ivar id: The unique ID of the VPC peering connection.
        :ivar accepter_vpc_info: Information on peer Vpc.
        :ivar requester_vpc_info: Information on requester Vpc.
        :ivar expiration_time: The expiration date and time for the VPC peering connection.
        :ivar status_code: The status of the VPC peering connection.
        :ivar status_message: A message that provides more information about the status of the VPC peering connection, if applicable.
        