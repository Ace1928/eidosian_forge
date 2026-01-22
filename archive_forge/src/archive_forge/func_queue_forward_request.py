import logging
import threading
def queue_forward_request(self, digest, whitelist=False):
    """If forwarding is enabled, insert a digest into the forwarding queue
        if whitelist is True, the digest will be forwarded as whitelist request
        if the queue is full, the digest is dropped
        """
    if self.forwarding_client is None:
        return
    try:
        self.forward_queue.put_nowait((digest, whitelist))
    except Queue.Full:
        pass