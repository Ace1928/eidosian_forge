import subprocess
import sys
import warnings
from argparse import ArgumentParser
from pathlib import Path
from packaging import version
from .. import AutoFeatureExtractor, AutoImageProcessor, AutoProcessor, AutoTokenizer
from ..utils import logging
from ..utils.import_utils import is_optimum_available
from .convert import export, validate_model_outputs
from .features import FeaturesManager
from .utils import get_preprocessor
def export_with_optimum(args):
    if is_optimum_available():
        from optimum.version import __version__ as optimum_version
        parsed_optimum_version = version.parse(optimum_version)
        if parsed_optimum_version < version.parse(MIN_OPTIMUM_VERSION):
            raise RuntimeError(f'transformers.onnx requires optimum >= {MIN_OPTIMUM_VERSION} but {optimum_version} is installed. You can upgrade optimum by running: pip install -U optimum[exporters]')
    else:
        raise RuntimeError('transformers.onnx requires optimum to run, you can install the library by running: pip install optimum[exporters]')
    cmd_line = [sys.executable, '-m', 'optimum.exporters.onnx', f'--model {args.model}', f'--task {args.feature}', f'--framework {args.framework}' if args.framework is not None else '', f'{args.output}']
    proc = subprocess.Popen(cmd_line, stdout=subprocess.PIPE)
    proc.wait()
    logger.info('The export was done by optimum.exporters.onnx. We recommend using to use this package directly in future, as transformers.onnx is deprecated, and will be removed in v5. You can find more information here: https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model.')