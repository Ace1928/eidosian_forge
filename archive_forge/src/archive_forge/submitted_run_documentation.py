import logging
import os
import signal
from abc import abstractmethod
from mlflow.entities import RunStatus
from mlflow.utils.annotations import developer_stable

        Cancel the run (interrupts the command subprocess, cancels the Databricks run, etc) and
        waits for it to terminate. The MLflow run status may not be set correctly
        upon run cancellation.
        