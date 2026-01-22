import time
from ase.utils.timing import Timer, timer
def test_timer():
    a = A()
    a.run()
    a.timer.write()
    t = a.timer.timers
    ty = t['run', 'yield']
    assert ty > 0.005, t