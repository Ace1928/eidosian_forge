import os
from celery import signals
from .base import BasePool, apply_target
Solo task pool (blocking, inline, fast).