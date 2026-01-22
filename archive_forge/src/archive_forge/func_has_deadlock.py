def has_deadlock(self):
    me = _thread.get_ident()
    tid = self.owner
    seen = set()
    while True:
        lock = _blocking_on.get(tid)
        if lock is None:
            return False
        tid = lock.owner
        if tid == me:
            return True
        if tid in seen:
            return False
        seen.add(tid)