from typing import Any, cast, Dict, Optional, TYPE_CHECKING, Union
import requests
import gitlab
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject, RESTObjectList
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .award_emojis import ProjectMergeRequestAwardEmojiManager  # noqa: F401
from .commits import ProjectCommit, ProjectCommitManager
from .discussions import ProjectMergeRequestDiscussionManager  # noqa: F401
from .draft_notes import ProjectMergeRequestDraftNoteManager
from .events import (  # noqa: F401
from .issues import ProjectIssue, ProjectIssueManager
from .merge_request_approvals import (  # noqa: F401
from .notes import ProjectMergeRequestNoteManager  # noqa: F401
from .pipelines import ProjectMergeRequestPipelineManager  # noqa: F401
from .reviewers import ProjectMergeRequestReviewerDetailManager
@cli.register_custom_action('ProjectMergeRequest')
@exc.on_http_error(exc.GitlabListError)
def closes_issues(self, **kwargs: Any) -> RESTObjectList:
    """List issues that will close on merge."

        Args:
            all: If True, return all the items, without pagination
            per_page: Number of items to retrieve per request
            page: ID of the page to return (starts with page 1)
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabListError: If the list could not be retrieved

        Returns:
            List of issues
        """
    path = f'{self.manager.path}/{self.encoded_id}/closes_issues'
    data_list = self.manager.gitlab.http_list(path, iterator=True, **kwargs)
    if TYPE_CHECKING:
        assert isinstance(data_list, gitlab.GitlabList)
    manager = ProjectIssueManager(self.manager.gitlab, parent=self.manager._parent)
    return RESTObjectList(manager, ProjectIssue, data_list)