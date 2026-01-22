from botocore.compat import six
from s3transfer.compat import accepts_kwargs
from s3transfer.exceptions import InvalidSubscriberMethodError
def on_queued(self, future, **kwargs):
    """Callback to be invoked when transfer request gets queued

        This callback can be useful for:

            * Keeping track of how many transfers have been requested
            * Providing the expected transfer size through
              future.meta.provide_transfer_size() so a HeadObject would not
              need to be made for copies and downloads.

        :type future: s3transfer.futures.TransferFuture
        :param future: The TransferFuture representing the requested transfer.
        """
    pass