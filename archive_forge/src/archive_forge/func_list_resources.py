from datetime import datetime
from boto.resultset import ResultSet
def list_resources(self, next_token=None):
    return self.connection.list_stack_resources(stack_name_or_id=self.stack_id, next_token=next_token)