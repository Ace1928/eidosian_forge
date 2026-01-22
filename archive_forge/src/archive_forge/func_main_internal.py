from __future__ import annotations
import os
import sys
import typing as t
from .init import (
from .constants import (
from .util import (
from .delegation import (
from .executor import (
from .timeout import (
from .data import (
from .util_common import (
from .cli import (
from .provisioning import (
from .config import (
def main_internal(cli_args: t.Optional[list[str]]=None) -> None:
    """Main program function."""
    try:
        os.chdir(data_context().content.root)
        args = parse_args(cli_args)
        config: CommonConfig = args.config(args)
        display.verbosity = config.verbosity
        display.truncate = config.truncate
        display.redact = config.redact
        display.color = config.color
        display.fd = sys.stderr if config.display_stderr else sys.stdout
        configure_timeout(config)
        report_locale(isinstance(config, TestConfig) and (not config.delegate))
        display.info('RLIMIT_NOFILE: %s' % (CURRENT_RLIMIT_NOFILE,), verbosity=2)
        delegate_args = None
        target_names = None
        try:
            if config.check_layout:
                data_context().check_layout()
            args.func(config)
        except PrimeContainers:
            pass
        except ListTargets as ex:
            target_names = ex.target_names
        except Delegate as ex:
            delegate_args = (ex.host_state, ex.exclude, ex.require)
        if delegate_args:
            delegate(config, *delegate_args)
        if target_names:
            for target_name in target_names:
                print(target_name)
        display.review_warnings()
        config.success = True
    except HostConnectionError as ex:
        display.fatal(str(ex))
        ex.run_callback()
        sys.exit(STATUS_HOST_CONNECTION_ERROR)
    except ApplicationWarning as ex:
        display.warning('%s' % ex)
        sys.exit(0)
    except ApplicationError as ex:
        display.fatal('%s' % ex)
        sys.exit(1)
    except TimeoutExpiredError as ex:
        display.fatal('%s' % ex)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(2)
    except BrokenPipeError:
        sys.exit(3)