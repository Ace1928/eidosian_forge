import abc
import collections
import logging
import os
import platform
import queue
import random
import sys
import threading
import time
import uuid
from oslo_utils import eventletutils
import proton
import pyngus
from oslo_messaging._drivers.amqp1_driver.addressing import AddresserFactory
from oslo_messaging._drivers.amqp1_driver.addressing import keyify
from oslo_messaging._drivers.amqp1_driver.addressing import SERVICE_NOTIFY
from oslo_messaging._drivers.amqp1_driver.addressing import SERVICE_RPC
from oslo_messaging._drivers.amqp1_driver import eventloop
from oslo_messaging import exceptions
from oslo_messaging.target import Target
from oslo_messaging import transport
def pyngus_callback(link, handle, state, info):
    if state == Sender._TIMED_OUT:
        return
    self._unacked.discard(send_task)
    if state == Sender._ACCEPTED:
        send_task._on_ack(Sender._ACCEPTED, info)
    elif state == Sender._RELEASED or (state == Sender._MODIFIED and (not info.get('delivery-failed')) and (not info.get('undeliverable-here'))):
        self._resend(send_task)
    else:
        send_task._on_ack(state, info)