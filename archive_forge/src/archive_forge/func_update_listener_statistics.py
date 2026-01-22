import os
import socket
import time
from oslo_serialization import jsonutils
import tenacity
from octavia_lib.api.drivers import data_models
from octavia_lib.api.drivers import exceptions as driver_exceptions
from octavia_lib.common import constants
def update_listener_statistics(self, statistics):
    """Update listener statistics.

        :param statistics: Statistics for listeners:
              id (string): ID for listener.
              active_connections (int): Number of currently active connections.
              bytes_in (int): Total bytes received.
              bytes_out (int): Total bytes sent.
              request_errors (int): Total requests not fulfilled.
              total_connections (int): The total connections handled.
        :type statistics: dict
        :raises: UpdateStatisticsError
        :returns: None
        """
    try:
        response = self._send(self.stats_socket, statistics)
    except Exception as e:
        raise driver_exceptions.UpdateStatisticsError(fault_string=str(e), stats_object=constants.LISTENERS)
    if response[constants.STATUS_CODE] != constants.DRVR_STATUS_CODE_OK:
        raise driver_exceptions.UpdateStatisticsError(fault_string=response.pop(constants.FAULT_STRING, None), stats_object=response.pop(constants.STATS_OBJECT, None), stats_object_id=response.pop(constants.STATS_OBJECT_ID, None), stats_record=response.pop(constants.STATS_RECORD, None))