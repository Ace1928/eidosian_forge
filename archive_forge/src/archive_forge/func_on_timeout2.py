import eventlet
from eventlet.hubs import get_hub
def on_timeout2():
    current.switch(([], [], []))