from __future__ import print_function
import os,stat,time
import errno
import sys
def hdict(self, hashroot):
    """ Get all data contained in hashed category 'hashroot' as dict """
    hfiles = self.keys(hashroot + '/*')
    hfiles.sort()
    last = len(hfiles) and hfiles[-1] or ''
    if last.endswith('xx'):
        hfiles = [last] + hfiles[:-1]
    all = {}
    for f in hfiles:
        try:
            all.update(self[f])
        except KeyError:
            print('Corrupt', f, 'deleted - hset is not threadsafe!')
            del self[f]
        self.uncache(f)
    return all