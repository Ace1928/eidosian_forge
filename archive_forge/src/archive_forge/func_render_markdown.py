from __future__ import annotations
import pickle
import urllib.parse
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, BinaryIO, TypeVar
import urllib3
from urllib3.util import Retry
import github.ApplicationOAuth
import github.Auth
import github.AuthenticatedUser
import github.Enterprise
import github.Event
import github.Gist
import github.GithubApp
import github.GithubIntegration
import github.GithubRetry
import github.GitignoreTemplate
import github.GlobalAdvisory
import github.License
import github.NamedUser
import github.Topic
from github import Consts
from github.GithubIntegration import GithubIntegration
from github.GithubObject import GithubObject, NotSet, Opt, is_defined
from github.GithubRetry import GithubRetry
from github.HookDelivery import HookDelivery, HookDeliverySummary
from github.HookDescription import HookDescription
from github.PaginatedList import PaginatedList
from github.RateLimit import RateLimit
from github.Requester import Requester
def render_markdown(self, text: str, context: Opt[Repository]=NotSet) -> str:
    """
        :calls: `POST /markdown <https://docs.github.com/en/rest/reference/markdown>`_
        :param text: string
        :param context: :class:`github.Repository.Repository`
        :rtype: string
        """
    assert isinstance(text, str), text
    assert context is NotSet or isinstance(context, github.Repository.Repository), context
    post_parameters = {'text': text}
    if is_defined(context):
        post_parameters['mode'] = 'gfm'
        post_parameters['context'] = context._identity
    status, headers, data = self.__requester.requestJson('POST', '/markdown', input=post_parameters)
    return data