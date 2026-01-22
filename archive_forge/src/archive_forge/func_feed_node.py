from contextlib import contextmanager
from typing import Dict, List
def feed_node(self, node):
    if self.is_issue(node):
        issue_node = self.get_node(node)
        self.add_issue(issue_node)