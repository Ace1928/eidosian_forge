from datetime import datetime
from boto.resultset import ResultSet
def describe_resource(self, logical_resource_id):
    return self.connection.describe_stack_resource(stack_name_or_id=self.stack_id, logical_resource_id=logical_resource_id)