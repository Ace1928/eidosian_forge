import functools
from dogpile.cache.backends import memcached as memcached_backend
from oslo_cache import _memcache_pool
from oslo_cache import exception
Memcached backend that does connection pooling.

    This memcached backend only allows for reuse of a client object,
    prevents too many client object from being instantiated, and maintains
    proper tracking of dead servers so as to limit delays when a server
    (or all servers) become unavailable.

    This backend doesn't allow to load balance things between servers.

    Memcached isn't HA. Values aren't automatically replicated between servers
    unless the client went out and wrote the value multiple time.

    The memcache server to use is determined by `python-memcached` itself by
    picking the host to use (from the given server list) based on a key hash.
    