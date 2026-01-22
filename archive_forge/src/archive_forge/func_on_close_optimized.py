import websocket
import threading
import logging
def on_close_optimized(ws):
    logging.info('WebSocket connection closed')