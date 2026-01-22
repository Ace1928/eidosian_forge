import abc
import logging
from six import with_metaclass
@classmethod
def registerEnv(cls, envCls):
    envName = envCls.__name__.lower().replace('yowsupenv', '')
    cls.__ENVS[envName] = envCls
    logger.debug('registered env %s => %s' % (envName, envCls))