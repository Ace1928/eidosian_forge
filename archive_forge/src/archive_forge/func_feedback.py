from ..Qt import QtCore, QtWidgets
def feedback(self, success, message=None, tip='', limitedTime=True):
    """Calls success() or failure(). If you want the message to be displayed until the user takes an action, set limitedTime to False. Then call self.reset() after the desired action.Threadsafe."""
    if success:
        self.success(message, tip, limitedTime=limitedTime)
    else:
        self.failure(message, tip, limitedTime=limitedTime)