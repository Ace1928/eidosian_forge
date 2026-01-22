import os
import os.path
import ssl
import threading
import unittest
import websocket as ws
@unittest.skipUnless(TEST_WITH_LOCAL_SERVER, 'Tests using local websocket server are disabled')
def testRunForeverTeardownCleanExit(self):
    """The WebSocketApp.run_forever() method should return `False` when the application ends gracefully."""
    app = ws.WebSocketApp(f'ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}')
    threading.Timer(interval=0.2, function=app.close).start()
    teardown = app.run_forever()
    self.assertEqual(teardown, False)