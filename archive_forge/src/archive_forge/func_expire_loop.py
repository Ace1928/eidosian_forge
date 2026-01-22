import logging
import sys
import time
import threading
from .kafka_metric import KafkaMetric
from .measurable import AnonMeasurable
from .metric_config import MetricConfig
from .metric_name import MetricName
from .stats import Sensor
def expire_loop():
    while True:
        time.sleep(30)
        self.ExpireSensorTask.run(self)