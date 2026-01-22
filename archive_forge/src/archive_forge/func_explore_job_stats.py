from __future__ import annotations
import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime
from functools import partial
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.utils.google import get_client_info
from langchain_community.vectorstores.utils import (
def explore_job_stats(self, job_id: str) -> Dict:
    """Return the statistics for a single job execution.

        Args:
            job_id: The BigQuery Job id.

        Returns:
            A dictionary of job statistics for a given job.
        """
    return self.bq_client.get_job(job_id)._properties['statistics']