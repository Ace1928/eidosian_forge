import logging
import os
import sys
import time
from collections import namedtuple
from warnings import warn
from billiard.einfo import ExceptionInfo, ExceptionWithTraceback
from kombu.exceptions import EncodeError
from kombu.serialization import loads as loads_message
from kombu.serialization import prepare_accept_content
from kombu.utils.encoding import safe_repr, safe_str
from celery import current_app, group, signals, states
from celery._state import _task_stack
from celery.app.task import Context
from celery.app.task import Task as BaseTask
from celery.exceptions import BackendGetMetaError, Ignore, InvalidTaskError, Reject, Retry
from celery.result import AsyncResult
from celery.utils.log import get_logger
from celery.utils.nodenames import gethostname
from celery.utils.objects import mro_lookup
from celery.utils.saferepr import saferepr
from celery.utils.serialization import get_pickleable_etype, get_pickleable_exception, get_pickled_exception
from celery.worker.state import successful_requests
def build_tracer(name, task, loader=None, hostname=None, store_errors=True, Info=TraceInfo, eager=False, propagate=False, app=None, monotonic=time.monotonic, trace_ok_t=trace_ok_t, IGNORE_STATES=IGNORE_STATES):
    """Return a function that traces task execution.

    Catches all exceptions and updates result backend with the
    state and result.

    If the call was successful, it saves the result to the task result
    backend, and sets the task status to `"SUCCESS"`.

    If the call raises :exc:`~@Retry`, it extracts
    the original exception, uses that as the result and sets the task state
    to `"RETRY"`.

    If the call results in an exception, it saves the exception as the task
    result, and sets the task state to `"FAILURE"`.

    Return a function that takes the following arguments:

        :param uuid: The id of the task.
        :param args: List of positional args to pass on to the function.
        :param kwargs: Keyword arguments mapping to pass on to the function.
        :keyword request: Request dict.

    """
    fun = task if task_has_custom(task, '__call__') else task.run
    loader = loader or app.loader
    ignore_result = task.ignore_result
    track_started = task.track_started
    track_started = not eager and (task.track_started and (not ignore_result))
    if eager and (not ignore_result) and task.store_eager_result:
        publish_result = True
    else:
        publish_result = not eager and (not ignore_result)
    deduplicate_successful_tasks = (app.conf.task_acks_late or task.acks_late) and app.conf.worker_deduplicate_successful_tasks and app.backend.persistent
    hostname = hostname or gethostname()
    inherit_parent_priority = app.conf.task_inherit_parent_priority
    loader_task_init = loader.on_task_init
    loader_cleanup = loader.on_process_cleanup
    task_before_start = None
    task_on_success = None
    task_after_return = None
    if task_has_custom(task, 'before_start'):
        task_before_start = task.before_start
    if task_has_custom(task, 'on_success'):
        task_on_success = task.on_success
    if task_has_custom(task, 'after_return'):
        task_after_return = task.after_return
    pid = os.getpid()
    request_stack = task.request_stack
    push_request = request_stack.push
    pop_request = request_stack.pop
    push_task = _task_stack.push
    pop_task = _task_stack.pop
    _does_info = logger.isEnabledFor(logging.INFO)
    resultrepr_maxsize = task.resultrepr_maxsize
    prerun_receivers = signals.task_prerun.receivers
    postrun_receivers = signals.task_postrun.receivers
    success_receivers = signals.task_success.receivers
    from celery import canvas
    signature = canvas.maybe_signature

    def on_error(request, exc, state=FAILURE, call_errbacks=True):
        if propagate:
            raise
        I = Info(state, exc)
        R = I.handle_error_state(task, request, eager=eager, call_errbacks=call_errbacks)
        return (I, R, I.state, I.retval)

    def trace_task(uuid, args, kwargs, request=None):
        R = I = T = Rstr = retval = state = None
        task_request = None
        time_start = monotonic()
        try:
            try:
                kwargs.items
            except AttributeError:
                raise InvalidTaskError('Task keyword arguments is not a mapping')
            task_request = Context(request or {}, args=args, called_directly=False, kwargs=kwargs)
            redelivered = task_request.delivery_info and task_request.delivery_info.get('redelivered', False)
            if deduplicate_successful_tasks and redelivered:
                if task_request.id in successful_requests:
                    return trace_ok_t(R, I, T, Rstr)
                r = AsyncResult(task_request.id, app=app)
                try:
                    state = r.state
                except BackendGetMetaError:
                    pass
                else:
                    if state == SUCCESS:
                        info(LOG_IGNORED, {'id': task_request.id, 'name': get_task_name(task_request, name), 'description': 'Task already completed successfully.'})
                        return trace_ok_t(R, I, T, Rstr)
            push_task(task)
            root_id = task_request.root_id or uuid
            task_priority = task_request.delivery_info.get('priority') if inherit_parent_priority else None
            push_request(task_request)
            try:
                if prerun_receivers:
                    send_prerun(sender=task, task_id=uuid, task=task, args=args, kwargs=kwargs)
                loader_task_init(uuid, task)
                if track_started:
                    task.backend.store_result(uuid, {'pid': pid, 'hostname': hostname}, STARTED, request=task_request)
                try:
                    if task_before_start:
                        task_before_start(uuid, args, kwargs)
                    R = retval = fun(*args, **kwargs)
                    state = SUCCESS
                except Reject as exc:
                    I, R = (Info(REJECTED, exc), ExceptionInfo(internal=True))
                    state, retval = (I.state, I.retval)
                    I.handle_reject(task, task_request)
                    traceback_clear(exc)
                except Ignore as exc:
                    I, R = (Info(IGNORED, exc), ExceptionInfo(internal=True))
                    state, retval = (I.state, I.retval)
                    I.handle_ignore(task, task_request)
                    traceback_clear(exc)
                except Retry as exc:
                    I, R, state, retval = on_error(task_request, exc, RETRY, call_errbacks=False)
                    traceback_clear(exc)
                except Exception as exc:
                    I, R, state, retval = on_error(task_request, exc)
                    traceback_clear(exc)
                except BaseException:
                    raise
                else:
                    try:
                        callbacks = task.request.callbacks
                        if callbacks:
                            if len(task.request.callbacks) > 1:
                                sigs, groups = ([], [])
                                for sig in callbacks:
                                    sig = signature(sig, app=app)
                                    if isinstance(sig, group):
                                        groups.append(sig)
                                    else:
                                        sigs.append(sig)
                                for group_ in groups:
                                    group_.apply_async((retval,), parent_id=uuid, root_id=root_id, priority=task_priority)
                                if sigs:
                                    group(sigs, app=app).apply_async((retval,), parent_id=uuid, root_id=root_id, priority=task_priority)
                            else:
                                signature(callbacks[0], app=app).apply_async((retval,), parent_id=uuid, root_id=root_id, priority=task_priority)
                        chain = task_request.chain
                        if chain:
                            _chsig = signature(chain.pop(), app=app)
                            _chsig.apply_async((retval,), chain=chain, parent_id=uuid, root_id=root_id, priority=task_priority)
                        task.backend.mark_as_done(uuid, retval, task_request, publish_result)
                    except EncodeError as exc:
                        I, R, state, retval = on_error(task_request, exc)
                    else:
                        Rstr = saferepr(R, resultrepr_maxsize)
                        T = monotonic() - time_start
                        if task_on_success:
                            task_on_success(retval, uuid, args, kwargs)
                        if success_receivers:
                            send_success(sender=task, result=retval)
                        if _does_info:
                            info(LOG_SUCCESS, {'id': uuid, 'name': get_task_name(task_request, name), 'return_value': Rstr, 'runtime': T, 'args': task_request.get('argsrepr') or safe_repr(args), 'kwargs': task_request.get('kwargsrepr') or safe_repr(kwargs)})
                if state not in IGNORE_STATES:
                    if task_after_return:
                        task_after_return(state, retval, uuid, args, kwargs, None)
            finally:
                try:
                    if postrun_receivers:
                        send_postrun(sender=task, task_id=uuid, task=task, args=args, kwargs=kwargs, retval=retval, state=state)
                finally:
                    pop_task()
                    pop_request()
                    if not eager:
                        try:
                            task.backend.process_cleanup()
                            loader_cleanup()
                        except (KeyboardInterrupt, SystemExit, MemoryError):
                            raise
                        except Exception as exc:
                            logger.error('Process cleanup failed: %r', exc, exc_info=True)
        except MemoryError:
            raise
        except Exception as exc:
            _signal_internal_error(task, uuid, args, kwargs, request, exc)
            if eager:
                raise
            R = report_internal_error(task, exc)
            if task_request is not None:
                I, _, _, _ = on_error(task_request, exc)
        return trace_ok_t(R, I, T, Rstr)
    return trace_task