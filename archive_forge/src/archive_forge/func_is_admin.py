import hashlib
import shutil
from pathlib import Path
from datetime import datetime
import asyncio
import aiofiles
import logging
from typing import Dict, List, Tuple, Union, Callable, Coroutine, Any, Optional
from functools import wraps
import threading
import ctypes
import sys
from PyQt5.QtWidgets import (
from PyQt5.QtCore import QDir, QThread, QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import QMainWindow
import os
import ctypes
def is_admin() -> bool:
    """
    Determines if the script is running with administrator or superuser privileges across different operating systems,
    including Windows, Linux, and Android. It ensures that on Android, the script does not run as admin unless the user
    has superuser privileges, which is generally not common. If not running with the required privileges, it prompts the
    user to relaunch the program with elevated privileges.

    Returns:
        bool: True if the script has administrator privileges or is running as root/superuser, False otherwise.
    """
    try:
        if os.name == 'nt':
            is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
            if not is_admin:
                logging.info('Requesting elevated privileges on Windows.')
                response = input('This script requires administrator privileges. Would you like to relaunch with elevated privileges? (y/n): ')
                if response.lower() == 'y':
                    ctypes.windll.shell32.ShellExecuteW(None, 'runas', sys.executable, ' '.join(sys.argv), None, 1)
                    sys.exit(0)
            return is_admin
        elif os.name == 'posix':
            is_root = os.geteuid() == 0
            if not is_root:
                logging.info('Requesting superuser privileges on POSIX systems.')
                response = input('This script requires superuser privileges. Would you like to attempt to relaunch with sudo? (y/n): ')
                if response.lower() == 'y':
                    os.execvp('sudo', ['sudo'] + sys.argv)
            return is_root
        elif os.name == 'android':
            is_superuser = os.system('su -c id') == 0
            if not is_superuser:
                logging.info('Superuser privileges are required on Android, but not available.')
            return is_superuser
        else:
            logging.info('Operating system not currently supported. Contact support for implementation.')
            return False
    except Exception as e:
        logging.error(f'Error checking admin privileges: {e}')
        return False