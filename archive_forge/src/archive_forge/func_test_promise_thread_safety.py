from promise import Promise
from promise.dataloader import DataLoader
import threading
def test_promise_thread_safety():
    """
    Promise tasks should never be executed in a different thread from the one they are scheduled from,
    unless the ThreadPoolExecutor is used.

    Here we assert that the pending promise tasks on thread 1 are not executed on thread 2 as thread 2 
    resolves its own promise tasks.
    """
    event_1 = threading.Event()
    event_2 = threading.Event()
    assert_object = {'is_same_thread': True}

    def task_1():
        thread_name = threading.current_thread().getName()

        def then_1(value):
            promise = Promise.resolve(None).then(then_2)
            assert promise.is_pending
            event_1.set()
            event_2.wait()

        def then_2(value):
            assert_object['is_same_thread'] = thread_name == threading.current_thread().getName()
        promise = Promise.resolve(None).then(then_1)

    def task_2():
        promise = Promise.resolve(None).then(lambda v: None)
        promise.get()
        event_2.set()
    thread_1 = threading.Thread(target=task_1)
    thread_1.start()
    event_1.wait()
    thread_2 = threading.Thread(target=task_2)
    thread_2.start()
    for thread in (thread_1, thread_2):
        thread.join()
    assert assert_object['is_same_thread']