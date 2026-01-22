from __future__ import annotations
import logging # isort:skip
from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import (
from ..util.dataclasses import (
 Takes over main program flow to perform the subcommand.

        *This method must be implemented by subclasses.*
        subclassed overwritten methods return different types:
        bool: Build
        None: FileOutput (subclassed by HTML, SVG and JSON. PNG overwrites FileOutput.invoke method), Info, Init,                 Sampledata, Secret, Serve, Static


        Args:
            args (argparse.Namespace) : command line arguments for the subcommand to parse

        Raises:
            NotImplementedError

        