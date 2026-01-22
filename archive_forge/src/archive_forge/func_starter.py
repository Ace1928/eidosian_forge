from typing import Generator
from simpy.core import Environment, SimTime
from simpy.events import Event, Process, ProcessGenerator
def starter() -> Generator[Event, None, Process]:
    yield env.timeout(delay)
    proc = env.process(generator)
    return proc