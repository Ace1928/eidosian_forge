import abc
import logging
from six import with_metaclass
@classmethod
def getEnv(cls, envName):
    if not envName in cls.__ENVS:
        raise ValueError('%s env does not exist' % envName)
    return cls.__ENVS[envName]()