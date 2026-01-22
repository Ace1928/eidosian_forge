from ._constants import *
def checklookbehindgroup(self, gid, source):
    if self.lookbehindgroups is not None:
        if not self.checkgroup(gid):
            raise source.error('cannot refer to an open group')
        if gid >= self.lookbehindgroups:
            raise source.error('cannot refer to group defined in the same lookbehind subpattern')