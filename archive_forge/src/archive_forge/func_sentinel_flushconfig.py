import warnings
def sentinel_flushconfig(self):
    """
        Force Sentinel to rewrite its configuration on disk, including the
        current Sentinel state.

        Normally Sentinel rewrites the configuration every time something
        changes in its state (in the context of the subset of the state which
        is persisted on disk across restart).
        However sometimes it is possible that the configuration file is lost
        because of operation errors, disk failures, package upgrade scripts or
        configuration managers. In those cases a way to to force Sentinel to
        rewrite the configuration file is handy.

        This command works even if the previous configuration file is
        completely missing.
        """
    return self.execute_command('SENTINEL FLUSHCONFIG')