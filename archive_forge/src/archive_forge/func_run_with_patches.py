def run_with_patches(self, f, *args, **kw):
    """Run 'f' with the given args and kwargs with all patches applied.

        Restores all objects to their original state when finished.
        """
    self.patch()
    try:
        return f(*args, **kw)
    finally:
        self.restore()