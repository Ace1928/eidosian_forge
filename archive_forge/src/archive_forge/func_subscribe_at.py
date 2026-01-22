from typing import Generator
from simpy.core import Environment, SimTime
from simpy.events import Event, Process, ProcessGenerator
def subscribe_at(event: Event) -> None:
    """Register at the *event* to receive an interrupt when it occurs.

    The most common use case for this is to pass
    a :class:`~simpy.events.Process` to get notified when it terminates.

    Raise a :exc:`RuntimeError` if ``event`` has already occurred.

    """
    env = event.env
    assert env.active_process is not None
    subscriber = env.active_process

    def signaller(signaller: Event, receiver: Process) -> ProcessGenerator:
        result = (yield signaller)
        if receiver.is_alive:
            receiver.interrupt((signaller, result))
    if event.callbacks is not None:
        env.process(signaller(event, subscriber))
    else:
        raise RuntimeError(f'{event} has already terminated.')