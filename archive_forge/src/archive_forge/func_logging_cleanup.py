from .. import fifo_cache, tests
def logging_cleanup(key, value):
    log.append((key, value))