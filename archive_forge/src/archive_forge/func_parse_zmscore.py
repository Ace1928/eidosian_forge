import datetime
from redis.utils import str_if_bytes
def parse_zmscore(response, **options):
    return [float(score) if score is not None else None for score in response]