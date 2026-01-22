import copy
from ansible_collections.amazon.aws.plugins.module_utils.retries import RetryingBotoClientWrapper
def rds_model(name):
    rds_models = core_waiter.WaiterModel(waiter_config=_inject_limit_retries(rds_data))
    return rds_models.get_waiter(name)