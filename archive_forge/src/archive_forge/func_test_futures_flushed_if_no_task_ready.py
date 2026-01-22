import duet
import duet.impl as impl
def test_futures_flushed_if_no_task_ready(self):
    future = CompleteOnFlush()
    task = make_task(future)
    rs = impl.ReadySet()
    rs.register(task)
    tasks = rs.get_all()
    assert tasks == [task]
    assert future.flushed