import logging
import sys
import time
import threading
from .kafka_metric import KafkaMetric
from .measurable import AnonMeasurable
from .metric_config import MetricConfig
from .metric_name import MetricName
from .stats import Sensor
def remove_sensor(self, name):
    """
        Remove a sensor (if it exists), associated metrics and its children.

        Arguments:
            name (str): The name of the sensor to be removed
        """
    sensor = self._sensors.get(name)
    if sensor:
        child_sensors = None
        with sensor._lock:
            with self._lock:
                val = self._sensors.pop(name, None)
                if val and val == sensor:
                    for metric in sensor.metrics:
                        self.remove_metric(metric.metric_name)
                    logger.debug('Removed sensor with name %s', name)
                    child_sensors = self._children_sensors.pop(sensor, None)
        if child_sensors:
            for child_sensor in child_sensors:
                self.remove_sensor(child_sensor.name)