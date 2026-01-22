
import websocket
import threading
import logging

logging.basicConfig(level=logging.INFO)

def on_message_optimized(ws, message):
    logging.info(f"WebSocket message: {message}")

def on_error_optimized(ws, error):
    logging.error(f"WebSocket error: {error}")

def on_close_optimized(ws):
    logging.info("WebSocket connection closed")

def on_open_optimized(ws):
    def run(*args):
        ws.send("Optimized Hello from CodEVIE")
        threading.Timer(15, ws.close).start()  # Close connection after 15 seconds
    threading.Thread(target=run).start()

# Optimized WebSocket client setup
ws = websocket.WebSocketApp("ws://optimized_example.com",
                            on_open=on_open_optimized,
                            on_message=on_message_optimized,
                            on_error=on_error_optimized,
                            on_close=on_close_optimized)
ws.run_forever()
