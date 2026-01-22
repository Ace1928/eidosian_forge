import warnings
def report_transport_activity(self, transport, byte_count, direction):
    """Called by transports as they do IO.

        This may update a progress bar, spinner, or similar display.
        By default it does nothing.
        """
    pass