def reactor_stop(*args):
    """Shutdown the twisted reactor main loop
        """
    if reactor.threadpool:
        Logger.info('Support: Stopping twisted threads')
        reactor.threadpool.stop()
    Logger.info('Support: Shutting down twisted reactor')
    reactor._mainLoopShutdown()
    try:
        reactor.stop()
    except ReactorNotRunning:
        pass
    import sys
    sys.modules.pop('twisted.internet.reactor', None)