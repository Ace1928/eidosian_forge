import torch._C._lazy
def metrics_report():
    """Return the combined (lazy core and backend) metric report"""
    return torch._C._lazy._metrics_report()