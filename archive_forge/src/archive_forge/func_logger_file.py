from __future__ import absolute_import
import urllib3
import copy
import logging
import multiprocessing
import sys
from six import iteritems
from six import with_metaclass
from six.moves import http_client as httplib
@logger_file.setter
def logger_file(self, value):
    """
        Sets the logger_file.

        If the logger_file is None, then add stream handler and remove file
        handler.
        Otherwise, add file handler and remove stream handler.

        :param value: The logger_file path.
        :type: str
        """
    self.__logger_file = value
    if self.__logger_file:
        self.logger_file_handler = logging.FileHandler(self.__logger_file)
        self.logger_file_handler.setFormatter(self.logger_formatter)
        for _, logger in iteritems(self.logger):
            logger.addHandler(self.logger_file_handler)
            if self.logger_stream_handler:
                logger.removeHandler(self.logger_stream_handler)
    else:
        self.logger_stream_handler = logging.StreamHandler()
        self.logger_stream_handler.setFormatter(self.logger_formatter)
        for _, logger in iteritems(self.logger):
            logger.addHandler(self.logger_stream_handler)
            if self.logger_file_handler:
                logger.removeHandler(self.logger_file_handler)