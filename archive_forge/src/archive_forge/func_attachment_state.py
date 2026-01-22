from boto.resultset import ResultSet
from boto.ec2.tag import Tag
from boto.ec2.ec2object import TaggedEC2Object
def attachment_state(self):
    """
        Get the attachment state.
        """
    state = None
    if self.attach_data:
        state = self.attach_data.status
    return state