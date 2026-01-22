import time
from boto.provider import Provider
def f_retry(*args, **kwargs):
    mtries, mdelay = (tries, delay)
    try_one_last_time = True
    while mtries > 1:
        try:
            return f(*args, **kwargs)
            try_one_last_time = False
            break
        except ExceptionToCheck as e:
            msg = '%s, Retrying in %d seconds...' % (str(e), mdelay)
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            time.sleep(mdelay)
            mtries -= 1
            mdelay *= backoff
    if try_one_last_time:
        return f(*args, **kwargs)
    return