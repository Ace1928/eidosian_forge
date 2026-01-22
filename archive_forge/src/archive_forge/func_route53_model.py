import copy
from ansible_collections.amazon.aws.plugins.module_utils.retries import RetryingBotoClientWrapper
def route53_model(name):
    route53_models = core_waiter.WaiterModel(waiter_config=_inject_limit_retries(route53_data))
    return route53_models.get_waiter(name)