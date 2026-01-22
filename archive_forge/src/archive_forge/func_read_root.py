import time
import argparse
from typing import Union, Sequence
import os
import sys
import psutil
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from utils.rwkv import *
from utils.torch import *
from utils.ngrok import *
from utils.log import log_middleware
from routes import completion, config, state_cache, midi, misc, file_process
import global_var
@app.get('/', tags=['Root'])
def read_root():
    return {'Hello': 'World!'}