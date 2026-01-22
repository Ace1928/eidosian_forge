import sys
import types
import stackless
def emulate():
    module = types.ModuleType('greenlet')
    sys.modules['greenlet'] = module
    module.greenlet = greenlet
    module.getcurrent = getcurrent
    module.GreenletExit = GreenletExit
    caller = stackless.getcurrent()
    tasklet_to_greenlet[caller] = None
    main_coro = greenlet()
    tasklet_to_greenlet[caller] = main_coro
    main_coro.t = caller
    del main_coro.switch
    coro_args[main_coro] = None