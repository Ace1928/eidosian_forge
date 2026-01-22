from __future__ import absolute_import, division, print_function
import time
def rediscache_to_dict(redis):
    result = dict(id=redis.id, name=redis.name, location=redis.location, sku=dict(name=redis.sku.name.lower(), size=redis.sku.family + str(redis.sku.capacity)), enable_non_ssl_port=redis.enable_non_ssl_port, host_name=redis.host_name, minimum_tls_version=redis.minimum_tls_version, public_network_access=redis.public_network_access, redis_version=redis.redis_version, shard_count=redis.shard_count, subnet=redis.subnet_id, static_ip=redis.static_ip, provisioning_state=redis.provisioning_state, tenant_settings=redis.tenant_settings, tags=redis.tags if redis.tags else None)
    for key in redis.redis_configuration:
        result[hyphen_to_underline(key)] = hyphen_to_underline(redis.redis_configuration.get(key, None))
    return result