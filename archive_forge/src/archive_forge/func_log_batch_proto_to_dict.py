from ray.core.generated.logging_pb2 import LogBatch
def log_batch_proto_to_dict(log_batch: LogBatch) -> dict:
    """Converts a LogBatch proto to a dict containing a batch of logs."""
    return {'ip': log_batch.ip, 'pid': log_batch.pid, 'job': log_batch.job_id, 'is_err': log_batch.is_error, 'lines': log_batch.lines, 'actor_name': log_batch.actor_name, 'task_name': log_batch.task_name}