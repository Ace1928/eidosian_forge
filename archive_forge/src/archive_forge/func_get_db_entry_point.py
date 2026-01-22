import os
from .start_hook_base import RayOnSparkStartHook
from .utils import get_spark_session
import logging
import threading
import time
def get_db_entry_point():
    """
    Return databricks entry_point instance, it is for calling some
    internal API in databricks runtime
    """
    from dbruntime import UserNamespaceInitializer
    user_namespace_initializer = UserNamespaceInitializer.getOrCreate()
    return user_namespace_initializer.get_spark_entry_point()