import json
import os
from flask import (
from flask_bootstrap import Bootstrap
from typing import Dict, Any, Optional, Union, Tuple, List, Callable
from nbconvert import HTMLExporter
import nbformat
import subprocess
import jupyterlab
import logging
from flask import request
import bcrypt
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.exc import IntegrityError

        Save configuration to a JSON file. 💾
        This method serializes the provided configuration dictionary into a JSON formatted string and meticulously writes it to the configuration file.
        It ensures that the configuration is persisted accurately and reliably, allowing the application to retain its state across sessions. 📅

        Args:
            config (Dict[str, Any]): A dictionary containing the configuration settings to be saved. 📦
        