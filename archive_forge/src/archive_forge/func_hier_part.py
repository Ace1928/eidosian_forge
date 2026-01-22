import re
@property
def hier_part(self):
    """The hierarchical part of the URI"""
    authority = self.authority
    if authority is None:
        return self.path
    else:
        return '//%s%s' % (authority, self.path)