from typing import Callable, Dict, Optional
from pydantic import BaseModel
from langsmith.evaluation.evaluator import EvaluationResult, RunEvaluator
from langsmith.schemas import Example, Run
Evaluate a single run.