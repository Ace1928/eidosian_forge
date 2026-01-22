from collections import namedtuple
from functools import partial
from threading import local
from .promise import Promise, async_instance, get_default_scheduler
def enqueue_post_promise_job(fn, scheduler):
    global cache
    if not hasattr(cache, 'resolved_promise'):
        cache.resolved_promise = Promise.resolve(None)
    if not scheduler:
        scheduler = get_default_scheduler()

    def on_promise_resolve(v):
        async_instance.invoke(fn, scheduler)
    cache.resolved_promise.then(on_promise_resolve)