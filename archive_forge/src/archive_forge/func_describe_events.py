from datetime import datetime
from boto.resultset import ResultSet
def describe_events(self, next_token=None):
    return self.connection.describe_stack_events(stack_name_or_id=self.stack_id, next_token=next_token)