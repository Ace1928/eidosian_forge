import logging
from . import Auth
from .AppAuthentication import AppAuthentication
from .GithubException import (
from .GithubIntegration import GithubIntegration
from .GithubRetry import GithubRetry
from .InputFileContent import InputFileContent
from .InputGitAuthor import InputGitAuthor
from .InputGitTreeElement import InputGitTreeElement
from .MainClass import Github
def set_log_level(level: int) -> None:
    """
    Set the log level of the github logger, e.g. set_log_level(logging.WARNING)
    :param level: log level
    """
    logger.setLevel(level)