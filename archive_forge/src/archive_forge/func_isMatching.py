import re
def isMatching(self, definition):
    if 'metadata' not in definition or 'labels' not in definition['metadata']:
        return False
    labels = definition['metadata']['labels']
    if not isinstance(labels, dict):
        return None
    return all((sel.isMatch(labels) for sel in self.selectors))