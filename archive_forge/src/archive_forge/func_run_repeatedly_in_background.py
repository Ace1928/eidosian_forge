import threading
from tensorboard import errors
def run_repeatedly_in_background(target, interval_sec):
    """Run a target task repeatedly in the background.

    In the context of this module, `target` is the `update()` method of the
    underlying reader for tfdbg2-format data.
    This method is mocked by unit tests for deterministic behaviors during
    testing.

    Args:
      target: The target task to run in the background, a callable with no args.
      interval_sec: Time interval between repeats, in seconds.

    Returns:
      - A `threading.Event` object that can be used to interrupt an ongoing
          waiting interval between successive runs of `target`. To interrupt the
          interval, call the `set()` method of the object.
      - The `threading.Thread` object on which `target` is run repeatedly.
    """
    event = threading.Event()

    def _run_repeatedly():
        while True:
            target()
            event.wait(interval_sec)
            event.clear()
    thread = threading.Thread(target=_run_repeatedly, daemon=True)
    thread.start()
    return (event, thread)