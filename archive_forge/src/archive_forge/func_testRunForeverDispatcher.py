import os
import os.path
import ssl
import threading
import unittest
import websocket as ws
@unittest.skipUnless(False, 'Test disabled for now (requires rel)')
def testRunForeverDispatcher(self):
    """A WebSocketApp should keep running as long as its self.keep_running
        is not False (in the boolean context).
        """

    def on_open(self, *args, **kwargs):
        """Send a message, receive, and send one more"""
        self.send('hello!')
        self.recv()
        self.send('goodbye!')

    def on_message(wsapp, message):
        print(message)
        self.close()
    app = ws.WebSocketApp(f'ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}', on_open=on_open, on_message=on_message)
    app.run_forever(dispatcher='Dispatcher')