import ipaddress
from functools import lru_cache
@lru_cache
def get_adapters_template(use_tz, timezone):
    ctx = adapt.AdaptersMap(adapters)
    ctx.register_loader('jsonb', TextLoader)
    ctx.register_loader('inet', TextLoader)
    ctx.register_loader('cidr', TextLoader)
    ctx.register_dumper(Range, DjangoRangeDumper)
    register_tzloader(timezone, ctx)
    return ctx