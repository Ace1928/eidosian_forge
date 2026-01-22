import torch._C
@property
def iters_per_second(self):
    """Return total number of iterations per second across all calling threads."""
    return self.num_iters / self.total_time_seconds