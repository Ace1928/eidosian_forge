def reactor_work(*args):
    """Process the twisted reactor task queue
        """
    Logger.trace('Support: processing twisted task queue')
    while len(q):
        q.popleft()()