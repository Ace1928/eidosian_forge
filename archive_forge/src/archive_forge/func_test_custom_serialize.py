import copy
import errno
import json
import os
import platform
import socket
import sys
import time
import warnings
from unittest import mock
import pytest
from pytest import mark
import zmq
from zmq.tests import BaseZMQTestCase, GreenTest, SkipTest, have_gevent, skip_pypy
def test_custom_serialize(self):
    a, b = self.create_bound_pair(zmq.DEALER, zmq.ROUTER)

    def serialize(msg):
        frames = []
        frames.extend(msg.get('identities', []))
        content = json.dumps(msg['content']).encode('utf8')
        frames.append(content)
        return frames

    def deserialize(frames):
        identities = frames[:-1]
        content = json.loads(frames[-1].decode('utf8'))
        return {'identities': identities, 'content': content}
    msg = {'content': {'a': 5, 'b': 'bee'}}
    a.send_serialized(msg, serialize)
    recvd = b.recv_serialized(deserialize)
    assert recvd['content'] == msg['content']
    assert recvd['identities']
    b.send_serialized(recvd, serialize)
    r2 = a.recv_serialized(deserialize)
    assert r2['content'] == msg['content']
    assert not r2['identities']