from concurrent import futures
import logging
Creates a thread pool that logs exceptions raised by the tasks within it.

    Args:
      max_workers: The maximum number of worker threads to allow the pool.

    Returns:
      A futures.ThreadPoolExecutor-compatible thread pool that logs exceptions
        raised by the tasks executed within it.
    