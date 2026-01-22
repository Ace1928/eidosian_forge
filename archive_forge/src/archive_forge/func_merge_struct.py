def merge_struct(self, reprocess=False):
    """Produce structured merge info"""
    struct_iter = self.iter_useful(self._merge_struct())
    if reprocess is True:
        return self.reprocess_struct(struct_iter)
    else:
        return struct_iter