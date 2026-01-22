import torch._C
@property
def latency_avg_ms(self):
    return self._c_stats.latency_avg_ms