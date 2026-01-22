from itertools import chain
from django.utils.inspect import func_accepts_kwargs
from django.utils.itercompat import is_iterable
def tags_available(self, deployment_checks=False):
    return set(chain.from_iterable((check.tags for check in self.get_checks(deployment_checks))))