import pprint
import click
from amqp import Connection, Message
from click_repl import register_repl
from celery.bin.base import handle_preload_options
def dump_message(message):
    if message is None:
        return 'No messages in queue. basic.publish something.'
    return {'body': message.body, 'properties': message.properties, 'delivery_info': message.delivery_info}