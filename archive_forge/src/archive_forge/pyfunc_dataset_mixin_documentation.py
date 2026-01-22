from abc import abstractmethod
from dataclasses import dataclass
from typing import List
from mlflow.models.evaluation.base import EvaluationDataset
from mlflow.models.utils import PyFuncInput, PyFuncOutput

        Converts the dataset to an EvaluationDataset for model evaluation.
        May not be implemented by all datasets.
        