def iter_entries_by_dir(self):
    """See Tree.iter_entries_by_dir."""
    for path, ie in self.oldtree.iter_entries_by_dir():
        yield (path, self.map_ie(ie))