from aiokafka.metrics.measurable_stat import AbstractMeasurableStat
from aiokafka.metrics.stats.sampled_stat import AbstractSampledStat
def unit_name(self):
    return TimeUnit.get_name(self._unit)