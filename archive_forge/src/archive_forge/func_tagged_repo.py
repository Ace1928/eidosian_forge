from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
@property
def tagged_repo(self):
    """Returns image repo string with tag, if it exists."""
    return '{0}:{1}'.format(self.repo, self.tag) if self.tag else self.repo