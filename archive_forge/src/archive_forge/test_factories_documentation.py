import pickle
from twisted.internet.protocol import Protocol, ReconnectingClientFactory
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase

                Record an attempt to reconnect, since this is what we
                are trying to avoid.
                