import os
import os.path
import ssl
import threading
import unittest
import websocket as ws
@unittest.skipUnless(TEST_WITH_LOCAL_SERVER, 'Tests using local websocket server are disabled')
def testCallbackFunctionException(self):
    """Test callback function exception handling"""
    exc = None
    passed_app = None

    def on_open(app):
        raise RuntimeError('Callback failed')

    def on_error(app, err):
        nonlocal passed_app
        passed_app = app
        nonlocal exc
        exc = err

    def on_pong(app, msg):
        app.close()
    app = ws.WebSocketApp(f'ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}', on_open=on_open, on_error=on_error, on_pong=on_pong)
    app.run_forever(ping_interval=2, ping_timeout=1)
    self.assertEqual(passed_app, app)
    self.assertIsInstance(exc, RuntimeError)
    self.assertEqual(str(exc), 'Callback failed')