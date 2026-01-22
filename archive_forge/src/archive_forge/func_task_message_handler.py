import logging
from kombu.asynchronous.timer import to_timestamp
from celery import signals
from celery.app import trace as _app_trace
from celery.exceptions import InvalidTaskError
from celery.utils.imports import symbol_by_name
from celery.utils.log import get_logger
from celery.utils.saferepr import saferepr
from celery.utils.time import timezone
from .request import create_request_cls
from .state import task_reserved
def task_message_handler(message, body, ack, reject, callbacks, to_timestamp=to_timestamp):
    if body is None and 'args' not in message.payload:
        body, headers, decoded, utc = (message.body, message.headers, False, app.uses_utc_timezone())
    elif 'args' in message.payload:
        body, headers, decoded, utc = hybrid_to_proto2(message, message.payload)
    else:
        body, headers, decoded, utc = proto1_to_proto2(message, body)
    req = Req(message, on_ack=ack, on_reject=reject, app=app, hostname=hostname, eventer=eventer, task=task, connection_errors=connection_errors, body=body, headers=headers, decoded=decoded, utc=utc)
    if _does_info:
        context = {'id': req.id, 'name': req.name, 'args': req.argsrepr, 'kwargs': req.kwargsrepr, 'eta': req.eta}
        info(_app_trace.LOG_RECEIVED, context, extra={'data': context})
    if (req.expires or req.id in revoked_tasks) and req.revoked():
        return
    signals.task_received.send(sender=consumer, request=req)
    if task_sends_events:
        send_event('task-received', uuid=req.id, name=req.name, args=req.argsrepr, kwargs=req.kwargsrepr, root_id=req.root_id, parent_id=req.parent_id, retries=req.request_dict.get('retries', 0), eta=req.eta and req.eta.isoformat(), expires=req.expires and req.expires.isoformat())
    bucket = None
    eta = None
    if req.eta:
        try:
            if req.utc:
                eta = to_timestamp(to_system_tz(req.eta))
            else:
                eta = to_timestamp(req.eta, app.timezone)
        except (OverflowError, ValueError) as exc:
            error("Couldn't convert ETA %r to timestamp: %r. Task: %r", req.eta, exc, req.info(safe=True), exc_info=True)
            req.reject(requeue=False)
    if rate_limits_enabled:
        bucket = get_bucket(task.name)
    if eta and bucket:
        consumer.qos.increment_eventually()
        return call_at(eta, limit_post_eta, (req, bucket, 1), priority=6)
    if eta:
        consumer.qos.increment_eventually()
        call_at(eta, apply_eta_task, (req,), priority=6)
        return task_message_handler
    if bucket:
        return limit_task(req, bucket, 1)
    task_reserved(req)
    if callbacks:
        [callback(req) for callback in callbacks]
    handle(req)