from twisted.application import app
from twisted.python.runtime import platformType
def runApp(config):
    runner = _SomeApplicationRunner(config)
    runner.run()
    if runner._exitSignal is not None:
        app._exitWithSignal(runner._exitSignal)