from __future__ import print_function
import os,stat,time
import errno
import sys
def hcompress(self, hashroot):
    """ Compress category 'hashroot', so hset is fast again

        hget will fail if fast_only is True for compressed items (that were
        hset before hcompress).

        """
    hfiles = self.keys(hashroot + '/*')
    all = {}
    for f in hfiles:
        all.update(self[f])
        self.uncache(f)
    self[hashroot + '/xx'] = all
    for f in hfiles:
        p = self.root / f
        if p.name == 'xx':
            continue
        p.unlink()