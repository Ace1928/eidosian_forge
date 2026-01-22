from snappy.snap import t3mlite as t3m
def check_loop(self, loop):
    l = len(loop)
    for i in range(l):
        s = loop[i].tet_and_perm_of_end()
        e = loop[(i + 1) % l].tet_and_perm
        if self.get_key(s) != self.get_key(e):
            raise Exception('Failed to be a loop at %d: %r' % (i, loop))