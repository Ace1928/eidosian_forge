import warnings
from typing import NoReturn, Set
from modin.logging import get_logger
from modin.utils import get_current_execution
@classmethod
def catch_bugs_and_request_email(cls, failure_condition: bool, extra_log: str='') -> None:
    if failure_condition:
        get_logger().info(f'Modin Error: Internal Error: {extra_log}')
        raise Exception('Internal Error. ' + 'Please visit https://github.com/modin-project/modin/issues ' + 'to file an issue with the traceback and the command that ' + "caused this error. If you can't file a GitHub issue, " + f'please email bug_reports@modin.org.\n{extra_log}')