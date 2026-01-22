from ._constants import *
def opengroup(self, name=None):
    gid = self.groups
    self.groupwidths.append(None)
    if self.groups > MAXGROUPS:
        raise error('too many groups')
    if name is not None:
        ogid = self.groupdict.get(name, None)
        if ogid is not None:
            raise error('redefinition of group name %r as group %d; was group %d' % (name, gid, ogid))
        self.groupdict[name] = gid
    return gid