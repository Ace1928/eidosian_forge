import asyncio
import logging
from typing import Tuple
import click
from ray import serve
from ray.serve._private.benchmarks.common import run_throughput_benchmark
from ray.serve.handle import DeploymentHandle, RayServeHandle
def hi(self) -> bytes:
    return b'hi'