from aiokafka.metrics.measurable import AnonMeasurable
from aiokafka.metrics.compound_stat import AbstractCompoundStat, NamedMeasurable
from .histogram import Histogram
from .sampled_stat import AbstractSampledStat
A compound stat that reports one or more percentiles