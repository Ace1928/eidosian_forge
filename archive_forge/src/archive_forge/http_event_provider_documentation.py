import asyncio
from typing import Dict
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import ray
from ray import serve
from ray.workflow import common, workflow_context, workflow_access
from ray.workflow.event_listener import EventListener
from ray.workflow.common import Event
import logging
workflow.wait_for_event calls this method after the event has
        been checkpointed and a transaction can be safely committed.