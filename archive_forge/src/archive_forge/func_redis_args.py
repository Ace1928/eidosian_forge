from typing import List
from redis import DataError
def redis_args(self):
    args = [self.name]
    if self.as_name:
        args += [self.AS, self.as_name]
    args += self.args
    args += self.args_suffix
    return args