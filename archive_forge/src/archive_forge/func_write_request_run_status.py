import logging
from typing import TYPE_CHECKING, Callable, Optional
from wandb.proto import wandb_internal_pb2 as pb
from wandb.proto import wandb_telemetry_pb2 as tpb
from ..interface.interface_queue import InterfaceQueue
from ..lib import proto_util, telemetry, tracelog
from . import context, datastore, flow_control
from .settings_static import SettingsStatic
def write_request_run_status(self, record: 'pb.Record') -> None:
    result = proto_util._result_from_record(record)
    if self._status_report:
        result.response.run_status_response.sync_time.CopyFrom(self._status_report.sync_time)
        send_record_num = self._status_report.record_num
        result.response.run_status_response.sync_items_total = self._record_num
        result.response.run_status_response.sync_items_pending = self._record_num - send_record_num
    self._respond_result(result)