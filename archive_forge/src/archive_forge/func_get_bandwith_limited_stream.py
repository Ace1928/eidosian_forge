import time
import threading
def get_bandwith_limited_stream(self, fileobj, transfer_coordinator, enabled=True):
    """Wraps a fileobj in a bandwidth limited stream wrapper

        :type fileobj: file-like obj
        :param fileobj: The file-like obj to wrap

        :type transfer_coordinator: s3transfer.futures.TransferCoordinator
        param transfer_coordinator: The coordinator for the general transfer
            that the wrapped stream is a part of

        :type enabled: boolean
        :param enabled: Whether bandwidth limiting should be enabled to start
        """
    stream = BandwidthLimitedStream(fileobj, self._leaky_bucket, transfer_coordinator, self._time_utils)
    if not enabled:
        stream.disable_bandwidth_limiting()
    return stream