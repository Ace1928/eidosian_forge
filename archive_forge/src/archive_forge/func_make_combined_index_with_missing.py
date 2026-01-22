from ... import errors, tests, transport
from .. import index as _mod_index
def make_combined_index_with_missing(self, missing=['1', '2']):
    """Create a CombinedGraphIndex which will have missing indexes.

        This creates a CGI which thinks it has 2 indexes, however they have
        been deleted. If CGI._reload_func() is called, then it will repopulate
        with a new index.

        :param missing: The underlying indexes to delete
        :return: (CombinedGraphIndex, reload_counter)
        """
    idx1 = self.make_index('1', nodes=[((b'1',), b'', ())])
    idx2 = self.make_index('2', nodes=[((b'2',), b'', ())])
    idx3 = self.make_index('3', nodes=[((b'1',), b'', ()), ((b'2',), b'', ())])
    reload_counter = [0, 0, 0]

    def reload():
        reload_counter[0] += 1
        new_indices = [idx3]
        if idx._indices == new_indices:
            reload_counter[2] += 1
            return False
        reload_counter[1] += 1
        idx._indices[:] = new_indices
        return True
    idx = _mod_index.CombinedGraphIndex([idx1, idx2], reload_func=reload)
    trans = self.get_transport()
    for fname in missing:
        trans.delete(fname)
    return (idx, reload_counter)