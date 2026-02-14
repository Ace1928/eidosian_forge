Hugging Face Hub API

Below is the documentation for the HfApi class, which serves as a Python wrapper for the Hugging Face Hub’s API.

All methods from the HfApi are also accessible from the package’s root directly, both approaches are detailed below.

The following approach uses the method from the root of the package:

from huggingface_hub import list_models

models = list_models()

The following approach uses the HfApi class:

from huggingface_hub import HfApi

hf_api = HfApi()
models = hf_api.list_models()

Using the HfApi class directly enables you to configure the client. In particular, a token can be passed to be authenticated in all API calls. This is different than huggingface-cli login or login() as the token is not persisted on the machine. One can also specify a different endpoint than the Hugging Face’s Hub (for example to interact with a Private Hub).

from huggingface_hub import HfApi

hf_api = HfApi(
    endpoint="https://huggingface.co", # Can be a Private Hub endpoint.
    token="hf_xxx", # Token is not persisted on the machine.
)

HfApi
class huggingface_hub.HfApi
< source >

( endpoint: typing.Optional[str] = Nonetoken: typing.Optional[str] = None )
change_discussion_status
< source >

( repo_id: strdiscussion_num: intnew_status: typing.Literal['open', 'closed']token: typing.Optional[str] = Nonecomment: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = None ) → DiscussionStatusChange

Parameters

repo_id (str) — A namespace (user or an organization) and a repo name separated by a /.
discussion_num (int) — The number of the Discussion or Pull Request . Must be a strictly positive integer.
new_status (str) — The new status for the discussion, either "open" or "closed".
comment (str, optional) — An optional comment to post with the status change.
repo_type (str, optional) — Set to "dataset" or "space" if uploading to a dataset or space, None or "model" if uploading to a model. Default is None.

    token (str, optional) — An authentication token (See https://huggingface.co/settings/token) 

Returns

DiscussionStatusChange

the status change event

Closes or re-opens a Discussion or Pull Request.

Examples:

new_title = "New title, fixing a typo"

HfApi().rename_discussion(

    repo_id="username/repo_name",

    discussion_num=34

    new_title=new_title

)
# DiscussionStatusChange(id='deadbeef0000000', type='status-change', ...)

Raises the following errors:

    HTTPError if the HuggingFace API returned an error
    ValueError if some parameter value is invalid
    RepositoryNotFoundError If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to private and you do not have access.

comment_discussion
< source >

( repo_id: strdiscussion_num: intcomment: strtoken: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = None ) → DiscussionComment

Parameters

repo_id (str) — A namespace (user or an organization) and a repo name separated by a /.
discussion_num (int) — The number of the Discussion or Pull Request . Must be a strictly positive integer.
comment (str) — The content of the comment to create. Comments support markdown formatting.
repo_type (str, optional) — Set to "dataset" or "space" if uploading to a dataset or space, None or "model" if uploading to a model. Default is None.

    token (str, optional) — An authentication token (See https://huggingface.co/settings/token) 

Returns

DiscussionComment

the newly created comment

Creates a new comment on the given Discussion.

Examples:


comment = """

Hello @otheruser!
...

# This is a title
...

**This is bold**, *this is italic* and ~this is strikethrough~

And [this](http://url) is a link

"""

HfApi().comment_discussion(

    repo_id="username/repo_name",

    discussion_num=34

    comment=comment

)
# DiscussionComment(id='deadbeef0000000', type='comment', ...)

Raises the following errors:

    HTTPError if the HuggingFace API returned an error
    ValueError if some parameter value is invalid
    RepositoryNotFoundError If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to private and you do not have access.

create_branch
< source >

( repo_id: strbranch: strtoken: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = None )

Parameters

repo_id (str) — The repository in which the branch will be created. Example: "user/my-cool-model".
branch (str) — The name of the branch to create.
token (str, optional) — Authentication token. Will default to the stored token.

    repo_type (str, optional) — Set to "dataset" or "space" if creating a branch on a dataset or space, None or "model" if tagging a model. Default is None. 

Raises

RepositoryNotFoundError or BadRequestError or HfHubHTTPError

    RepositoryNotFoundError — If repository is not found (error 404): wrong repo_id/repo_type, private but not authenticated or repo does not exist.
    BadRequestError — If invalid reference for a branch. Ex: refs/pr/5 or ‘refs/foo/bar’.
    HfHubHTTPError — If the branch already exists on the repo (error 409).

Create a new branch from main on a repo on the Hub.
create_commit
< source >

( repo_id: stroperations: typing.Iterable[typing.Union[huggingface_hub._commit_api.CommitOperationAdd, huggingface_hub._commit_api.CommitOperationDelete]]commit_message: strcommit_description: typing.Optional[str] = Nonetoken: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = Nonerevision: typing.Optional[str] = Nonecreate_pr: typing.Optional[bool] = Nonenum_threads: int = 5parent_commit: typing.Optional[str] = None ) → CommitInfo

Parameters

repo_id (str) — The repository in which the commit will be created, for example: "username/custom_transformers"
operations (Iterable of CommitOperation()) — An iterable of operations to include in the commit, either:

    CommitOperationAdd to upload a file
    CommitOperationDelete to delete a file

commit_message (str) — The summary (first line) of the commit that will be created.
commit_description (str, optional) — The description of the commit that will be created
token (str, optional) — Authentication token, obtained with HfApi.login method. Will default to the stored token.
repo_type (str, optional) — Set to "dataset" or "space" if uploading to a dataset or space, None or "model" if uploading to a model. Default is None.
revision (str, optional) — The git revision to commit from. Defaults to the head of the "main" branch.
create_pr (boolean, optional) — Whether or not to create a Pull Request with that commit. Defaults to False. If revision is not set, PR is opened against the "main" branch. If revision is set and is a branch, PR is opened against this branch. If revision is set and is not a branch name (example: a commit oid), an RevisionNotFoundError is returned by the server.
num_threads (int, optional) — Number of concurrent threads for uploading files. Defaults to 5. Setting it to 2 means at most 2 files will be uploaded concurrently.

    parent_commit (str, optional) — The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported.If specified and create_pr is False, the commit will fail if revision does not point to parent_commit. If specified and create_pr is True, the pull request will be created from parent_commit. Specifying parent_commit ensures the repo has not changed before committing the changes, and can be especially useful if the repo is updated / committed to concurrently. 

Returns

CommitInfo

Instance of CommitInfo containing information about the newly created commit (commit hash, commit url, pr url, commit message,…).

Raises

ValueError or RepositoryNotFoundError

    ValueError — If commit message is empty.
    ValueError — If parent commit is not a valid commit OID.
    ValueError — If the Hub API returns an HTTP 400 error (bad request)
    ValueError — If create_pr is True and revision is neither None nor "main".
    RepositoryNotFoundError — If repository is not found (error 404): wrong repo_id/repo_type, private but not authenticated or repo does not exist.

Creates a commit in the given repo, deleting & uploading files as needed.

create_commit assumes that the repo already exists on the Hub. If you get a Client error 404, please make sure you are authenticated and that repo_id and repo_type are set correctly. If repo does not exist, create it first using create_repo().

create_commit is limited to 25k LFS files and a 1GB payload for regular files.
create_discussion
< source >

( repo_id: strtitle: strtoken: typing.Optional[str] = Nonedescription: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = Nonepull_request: bool = False )

Parameters

repo_id (str) — A namespace (user or an organization) and a repo name separated by a /.
title (str) — The title of the discussion. It can be up to 200 characters long, and must be at least 3 characters long. Leading and trailing whitespaces will be stripped.
token (str, optional) — An authentication token (See https://huggingface.co/settings/token)
description (str, optional) — An optional description for the Pull Request. Defaults to "Discussion opened with the huggingface_hub Python library"
pull_request (bool, optional) — Whether to create a Pull Request or discussion. If True, creates a Pull Request. If False, creates a discussion. Defaults to False.

    repo_type (str, optional) — Set to "dataset" or "space" if uploading to a dataset or space, None or "model" if uploading to a model. Default is None. 

Creates a Discussion or Pull Request.

Pull Requests created programmatically will be in "draft" status.

Creating a Pull Request with changes can also be done at once with HfApi.create_commit().

Returns: DiscussionWithDetails

Raises the following errors:

    HTTPError if the HuggingFace API returned an error
    ValueError if some parameter value is invalid
    RepositoryNotFoundError If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to private and you do not have access.

create_pull_request
< source >

( repo_id: strtitle: strtoken: typing.Optional[str] = Nonedescription: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = None )

Parameters

repo_id (str) — A namespace (user or an organization) and a repo name separated by a /.
title (str) — The title of the discussion. It can be up to 200 characters long, and must be at least 3 characters long. Leading and trailing whitespaces will be stripped.
token (str, optional) — An authentication token (See https://huggingface.co/settings/token)
description (str, optional) — An optional description for the Pull Request. Defaults to "Discussion opened with the huggingface_hub Python library"

    repo_type (str, optional) — Set to "dataset" or "space" if uploading to a dataset or space, None or "model" if uploading to a model. Default is None. 

Creates a Pull Request . Pull Requests created programmatically will be in "draft" status.

Creating a Pull Request with changes can also be done at once with HfApi.create_commit();

This is a wrapper around HfApi.create_discusssion.

Returns: DiscussionWithDetails

Raises the following errors:

    HTTPError if the HuggingFace API returned an error
    ValueError if some parameter value is invalid
    RepositoryNotFoundError If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to private and you do not have access.

create_repo
< source >

( repo_id: strtoken: typing.Optional[str] = Noneprivate: bool = Falserepo_type: typing.Optional[str] = Noneexist_ok: bool = Falsespace_sdk: typing.Optional[str] = None ) → str

Parameters

repo_id (str) — A namespace (user or an organization) and a repo name separated by a /.
token (str, optional) — An authentication token (See https://huggingface.co/settings/token)
private (bool, optional, defaults to False) — Whether the model repo should be private.
repo_type (str, optional) — Set to "dataset" or "space" if uploading to a dataset or space, None or "model" if uploading to a model. Default is None.
exist_ok (bool, optional, defaults to False) — If True, do not raise an error if repo already exists.

    space_sdk (str, optional) — Choice of SDK to use if repo_type is “space”. Can be “streamlit”, “gradio”, or “static”. 

Returns

str

URL to the newly created repo.

Create an empty repo on the HuggingFace Hub.
create_tag
< source >

( repo_id: strtag: strtag_message: typing.Optional[str] = Nonerevision: typing.Optional[str] = Nonetoken: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = None )

Parameters

repo_id (str) — The repository in which a commit will be tagged. Example: "user/my-cool-model".
tag (str) — The name of the tag to create.
tag_message (str, optional) — The description of the tag to create.
revision (str, optional) — The git revision to tag. It can be a branch name or the OID/SHA of a commit, as a hexadecimal string. Shorthands (7 first characters) are also supported. Defaults to the head of the "main" branch.
token (str, optional) — Authentication token. Will default to the stored token.

    repo_type (str, optional) — Set to "dataset" or "space" if tagging a dataset or space, None or "model" if tagging a model. Default is None. 

Raises

RepositoryNotFoundError or RevisionNotFoundError

    RepositoryNotFoundError — If repository is not found (error 404): wrong repo_id/repo_type, private but not authenticated or repo does not exist.
    RevisionNotFoundError — If revision is not found (error 404) on the repo.

Tag a given commit of a repo on the Hub.
dataset_info
< source >

( repo_id: strrevision: typing.Optional[str] = Nonetimeout: typing.Optional[float] = Nonefiles_metadata: bool = Falsetoken: typing.Union[bool, str, NoneType] = None ) → hf_api.DatasetInfo

Parameters

repo_id (str) — A namespace (user or an organization) and a repo name separated by a /.
revision (str, optional) — The revision of the dataset repository from which to get the information.
timeout (float, optional) — Whether to set a timeout for the request to the Hub.
files_metadata (bool, optional) — Whether or not to retrieve metadata for files in the repository (size, LFS metadata, etc). Defaults to False.

    token (bool or str, optional) — A valid authentication token (see https://huggingface.co/settings/token). If None or True and machine is logged in (through huggingface-cli login or login()), token will be retrieved from the cache. If False, token is not sent in the request header. 

Returns

hf_api.DatasetInfo

The dataset repository information.

Get info on one specific dataset on huggingface.co.

Dataset can be private if you pass an acceptable token.

Raises the following errors:

    RepositoryNotFoundError If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to private and you do not have access.
    RevisionNotFoundError If the revision to download from cannot be found.

delete_branch
< source >

( repo_id: strbranch: strtoken: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = None )

Parameters

repo_id (str) — The repository in which a branch will be deleted. Example: "user/my-cool-model".
branch (str) — The name of the branch to delete.
token (str, optional) — Authentication token. Will default to the stored token.

    repo_type (str, optional) — Set to "dataset" or "space" if creating a branch on a dataset or space, None or "model" if tagging a model. Default is None. 

Raises

RepositoryNotFoundError or HfHubHTTPError

    RepositoryNotFoundError — If repository is not found (error 404): wrong repo_id/repo_type, private but not authenticated or repo does not exist.
    HfHubHTTPError — If trying to delete a protected branch. Ex: main cannot be deleted.
    HfHubHTTPError — If trying to delete a branch that does not exist.

Delete a branch from a repo on the Hub.
delete_file
< source >

( path_in_repo: strrepo_id: strtoken: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = Nonerevision: typing.Optional[str] = Nonecommit_message: typing.Optional[str] = Nonecommit_description: typing.Optional[str] = Nonecreate_pr: typing.Optional[bool] = Noneparent_commit: typing.Optional[str] = None )

Parameters

path_in_repo (str) — Relative filepath in the repo, for example: "checkpoints/1fec34a/weights.bin"
repo_id (str) — The repository from which the file will be deleted, for example: "username/custom_transformers"
token (str, optional) — Authentication token, obtained with HfApi.login method. Will default to the stored token.
repo_type (str, optional) — Set to "dataset" or "space" if the file is in a dataset or space, None or "model" if in a model. Default is None.
revision (str, optional) — The git revision to commit from. Defaults to the head of the "main" branch.
commit_message (str, optional) — The summary / title / first line of the generated commit. Defaults to f"Delete {path_in_repo} with huggingface_hub".
commit_description (str optional) — The description of the generated commit
create_pr (boolean, optional) — Whether or not to create a Pull Request with that commit. Defaults to False. If revision is not set, PR is opened against the "main" branch. If revision is set and is a branch, PR is opened against this branch. If revision is set and is not a branch name (example: a commit oid), an RevisionNotFoundError is returned by the server.

    parent_commit (str, optional) — The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported. If specified and create_pr is False, the commit will fail if revision does not point to parent_commit. If specified and create_pr is True, the pull request will be created from parent_commit. Specifying parent_commit ensures the repo has not changed before committing the changes, and can be especially useful if the repo is updated / committed to concurrently. 

Deletes a file in the given repo.

Raises the following errors:

    HTTPError if the HuggingFace API returned an error
    ValueError if some parameter value is invalid
    RepositoryNotFoundError If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to private and you do not have access.
    RevisionNotFoundError If the revision to download from cannot be found.
    EntryNotFoundError If the file to download cannot be found.

delete_folder
< source >

( path_in_repo: strrepo_id: strtoken: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = Nonerevision: typing.Optional[str] = Nonecommit_message: typing.Optional[str] = Nonecommit_description: typing.Optional[str] = Nonecreate_pr: typing.Optional[bool] = Noneparent_commit: typing.Optional[str] = None )

Parameters

path_in_repo (str) — Relative folder path in the repo, for example: "checkpoints/1fec34a".
repo_id (str) — The repository from which the folder will be deleted, for example: "username/custom_transformers"
token (str, optional) — Authentication token, obtained with HfApi.login method. Will default to the stored token.
repo_type (str, optional) — Set to "dataset" or "space" if the folder is in a dataset or space, None or "model" if in a model. Default is None.
revision (str, optional) — The git revision to commit from. Defaults to the head of the "main" branch.
commit_message (str, optional) — The summary / title / first line of the generated commit. Defaults to f"Delete folder {path_in_repo} with huggingface_hub".
commit_description (str optional) — The description of the generated commit.
create_pr (boolean, optional) — Whether or not to create a Pull Request with that commit. Defaults to False. If revision is not set, PR is opened against the "main" branch. If revision is set and is a branch, PR is opened against this branch. If revision is set and is not a branch name (example: a commit oid), an RevisionNotFoundError is returned by the server.

    parent_commit (str, optional) — The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported. If specified and create_pr is False, the commit will fail if revision does not point to parent_commit. If specified and create_pr is True, the pull request will be created from parent_commit. Specifying parent_commit ensures the repo has not changed before committing the changes, and can be especially useful if the repo is updated / committed to concurrently. 

Deletes a folder in the given repo.

Simple wrapper around create_commit() method.
delete_repo
< source >

( repo_id: strtoken: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = None )

Parameters

repo_id (str) — A namespace (user or an organization) and a repo name separated by a /.
token (str, optional) — An authentication token (See https://huggingface.co/settings/token)

    repo_type (str, optional) — Set to "dataset" or "space" if uploading to a dataset or space, None or "model" if uploading to a model. 

Delete a repo from the HuggingFace Hub. CAUTION: this is irreversible.

Raises the following errors:

    RepositoryNotFoundError If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to private and you do not have access.

delete_tag
< source >

( repo_id: strtag: strtoken: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = None )

Parameters

repo_id (str) — The repository in which a tag will be deleted. Example: "user/my-cool-model".
tag (str) — The name of the tag to delete.
token (str, optional) — Authentication token. Will default to the stored token.

    repo_type (str, optional) — Set to "dataset" or "space" if tagging a dataset or space, None or "model" if tagging a model. Default is None. 

Raises

RepositoryNotFoundError or RevisionNotFoundError

    RepositoryNotFoundError — If repository is not found (error 404): wrong repo_id/repo_type, private but not authenticated or repo does not exist.
    RevisionNotFoundError — If tag is not found.

Delete a tag from a repo on the Hub.
edit_discussion_comment
< source >

( repo_id: strdiscussion_num: intcomment_id: strnew_content: strtoken: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = None ) → DiscussionComment

Parameters

repo_id (str) — A namespace (user or an organization) and a repo name separated by a /.
discussion_num (int) — The number of the Discussion or Pull Request . Must be a strictly positive integer.
comment_id (str) — The ID of the comment to edit.
new_content (str) — The new content of the comment. Comments support markdown formatting.
repo_type (str, optional) — Set to "dataset" or "space" if uploading to a dataset or space, None or "model" if uploading to a model. Default is None.

    token (str, optional) — An authentication token (See https://huggingface.co/settings/token) 

Returns

DiscussionComment

the edited comment

Edits a comment on a Discussion / Pull Request.

Raises the following errors:

    HTTPError if the HuggingFace API returned an error
    ValueError if some parameter value is invalid
    RepositoryNotFoundError If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to private and you do not have access.

get_dataset_tags
< source >

( )

Gets all valid dataset tags as a nested namespace object.
get_discussion_details
< source >

( repo_id: strdiscussion_num: intrepo_type: typing.Optional[str] = Nonetoken: typing.Optional[str] = None )

Parameters

repo_id (str) — A namespace (user or an organization) and a repo name separated by a /.
discussion_num (int) — The number of the Discussion or Pull Request . Must be a strictly positive integer.
repo_type (str, optional) — Set to "dataset" or "space" if uploading to a dataset or space, None or "model" if uploading to a model. Default is None.

    token (str, optional) — An authentication token (See https://huggingface.co/settings/token) 

Fetches a Discussion’s / Pull Request ‘s details from the Hub.

Returns: DiscussionWithDetails

Raises the following errors:

    HTTPError if the HuggingFace API returned an error
    ValueError if some parameter value is invalid
    RepositoryNotFoundError If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to private and you do not have access.

get_full_repo_name
< source >

( model_id: strorganization: typing.Optional[str] = Nonetoken: typing.Union[bool, str, NoneType] = None ) → str

Parameters

model_id (str) — The name of the model.
organization (str, optional) — If passed, the repository name will be in the organization namespace instead of the user namespace.

    token (bool or str, optional) — A valid authentication token (see https://huggingface.co/settings/token). If None or True and machine is logged in (through huggingface-cli login or login()), token will be retrieved from the cache. If False, token is not sent in the request header. 

Returns

str

The repository name in the user’s namespace ({username}/{model_id}) if no organization is passed, and under the organization namespace ({organization}/{model_id}) otherwise.

Returns the repository name for a given model ID and optional organization.
get_model_tags
< source >

( )

Gets all valid model tags as a nested namespace object
get_repo_discussions
< source >

( repo_id: strrepo_type: typing.Optional[str] = Nonetoken: typing.Optional[str] = None ) → Iterator[Discussion]

Parameters

repo_id (str) — A namespace (user or an organization) and a repo name separated by a /.
repo_type (str, optional) — Set to "dataset" or "space" if fetching from a dataset or space, None or "model" if fetching from a model. Default is None.

    token (str, optional) — An authentication token (See https://huggingface.co/settings/token). 

Returns

Iterator[Discussion]

An iterator of Discussion objects.

Fetches Discussions and Pull Requests for the given repo.

Example:

Collecting all discussions of a repo in a list:

from huggingface_hub import get_repo_discussions

discussions_list = list(get_repo_discussions(repo_id="bert-base-uncased"))

Iterating over discussions of a repo:

from huggingface_hub import get_repo_discussions

for discussion in get_repo_discussions(repo_id="bert-base-uncased"):

    print(discussion.num, discussion.title)

hide_discussion_comment
< source >

( repo_id: strdiscussion_num: intcomment_id: strtoken: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = None ) → DiscussionComment

Parameters

repo_id (str) — A namespace (user or an organization) and a repo name separated by a /.
discussion_num (int) — The number of the Discussion or Pull Request . Must be a strictly positive integer.
comment_id (str) — The ID of the comment to edit.
repo_type (str, optional) — Set to "dataset" or "space" if uploading to a dataset or space, None or "model" if uploading to a model. Default is None.

    token (str, optional) — An authentication token (See https://huggingface.co/settings/token) 

Returns

DiscussionComment

the hidden comment

Hides a comment on a Discussion / Pull Request.
Hidden comments' content cannot be retrieved anymore. Hiding a comment is irreversible.

Raises the following errors:

    HTTPError if the HuggingFace API returned an error
    ValueError if some parameter value is invalid
    RepositoryNotFoundError If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to private and you do not have access.

list_datasets
< source >

( filter: typing.Union[huggingface_hub.utils.endpoint_helpers.DatasetFilter, str, typing.Iterable[str], NoneType] = Noneauthor: typing.Optional[str] = Nonesearch: typing.Optional[str] = Nonesort: typing.Union[typing.Literal['lastModified'], str, NoneType] = Nonedirection: typing.Optional[typing.Literal[-1]] = Nonelimit: typing.Optional[int] = NonecardData: typing.Optional[bool] = Nonefull: typing.Optional[bool] = Nonetoken: typing.Optional[str] = None ) → List[DatasetInfo]

Parameters

filter (DatasetFilter or str or Iterable, optional) — A string or DatasetFilter which can be used to identify datasets on the hub.
author (str, optional) — A string which identify the author of the returned datasets.
search (str, optional) — A string that will be contained in the returned datasets.
sort (Literal["lastModified"] or str, optional) — The key with which to sort the resulting datasets. Possible values are the properties of the huggingface_hub.hf_api.DatasetInfo class.
direction (Literal[-1] or int, optional) — Direction in which to sort. The value -1 sorts by descending order while all other values sort by ascending order.
limit (int, optional) — The limit on the number of datasets fetched. Leaving this option to None fetches all datasets.
full (bool, optional) — Whether to fetch all dataset data, including the lastModified and the cardData. Can contain useful information such as the PapersWithCode ID.

    token (bool or str, optional) — A valid authentication token (see https://huggingface.co/settings/token). If None or True and machine is logged in (through huggingface-cli login or login()), token will be retrieved from the cache. If False, token is not sent in the request header. 

Returns

List[DatasetInfo]

a list of huggingface_hub.hf_api.DatasetInfo objects. To anticipate future pagination, please consider the return value to be a simple iterator.

Get the list of all the datasets on huggingface.co

Example usage with the filter argument:

from huggingface_hub import HfApi

api = HfApi()

# List all datasets

api.list_datasets()

# Get all valid search arguments

args = DatasetSearchArguments()

# List only the text classification datasets

api.list_datasets(filter="task_categories:text-classification")

# Using the `DatasetFilter`

filt = DatasetFilter(task_categories="text-classification")

# With `DatasetSearchArguments`

filt = DatasetFilter(task=args.task_categories.text_classification)

api.list_models(filter=filt)

# List only the datasets in russian for language modeling

api.list_datasets(

    filter=("language:ru", "task_ids:language-modeling")

)

# Using the `DatasetFilter`

filt = DatasetFilter(language="ru", task_ids="language-modeling")

# With `DatasetSearchArguments`

filt = DatasetFilter(

    language=args.language.ru,

    task_ids=args.task_ids.language_modeling,

)

api.list_datasets(filter=filt)

Example usage with the search argument:

from huggingface_hub import HfApi

api = HfApi()

# List all datasets with "text" in their name

api.list_datasets(search="text")

# List all datasets with "text" in their name made by google

api.list_datasets(search="text", author="google")

list_metrics
< source >

( ) → List[MetricInfo]

Returns

List[MetricInfo]

a list of MetricInfo objects which.

Get the public list of all the metrics on huggingface.co
list_models
< source >

( filter: typing.Union[huggingface_hub.utils.endpoint_helpers.ModelFilter, str, typing.Iterable[str], NoneType] = Noneauthor: typing.Optional[str] = Nonesearch: typing.Optional[str] = Noneemissions_thresholds: typing.Union[typing.Tuple[float, float], NoneType] = Nonesort: typing.Union[typing.Literal['lastModified'], str, NoneType] = Nonedirection: typing.Optional[typing.Literal[-1]] = Nonelimit: typing.Optional[int] = Nonefull: typing.Optional[bool] = NonecardData: bool = Falsefetch_config: bool = Falsetoken: typing.Union[bool, str, NoneType] = None ) → List[ModelInfo]

Parameters

filter (ModelFilter or str or Iterable, optional) — A string or ModelFilter which can be used to identify models on the Hub.
author (str, optional) — A string which identify the author (user or organization) of the returned models
search (str, optional) — A string that will be contained in the returned models Example usage:
emissions_thresholds (Tuple, optional) — A tuple of two ints or floats representing a minimum and maximum carbon footprint to filter the resulting models with in grams.
sort (Literal["lastModified"] or str, optional) — The key with which to sort the resulting models. Possible values are the properties of the huggingface_hub.hf_api.ModelInfo class.
direction (Literal[-1] or int, optional) — Direction in which to sort. The value -1 sorts by descending order while all other values sort by ascending order.
limit (int, optional) — The limit on the number of models fetched. Leaving this option to None fetches all models.
full (bool, optional) — Whether to fetch all model data, including the lastModified, the sha, the files and the tags. This is set to True by default when using a filter.
cardData (bool, optional) — Whether to grab the metadata for the model as well. Can contain useful information such as carbon emissions, metrics, and datasets trained on.
fetch_config (bool, optional) — Whether to fetch the model configs as well. This is not included in full due to its size.

    token (bool or str, optional) — A valid authentication token (see https://huggingface.co/settings/token). If None or True and machine is logged in (through huggingface-cli login or login()), token will be retrieved from the cache. If False, token is not sent in the request header. 

Returns

List[ModelInfo]

a list of huggingface_hub.hf_api.ModelInfo objects. To anticipate future pagination, please consider the return value to be a simple iterator.

Get the list of all the models on huggingface.co

Example usage with the filter argument:

from huggingface_hub import HfApi

api = HfApi()

# List all models

api.list_models()

# Get all valid search arguments

args = ModelSearchArguments()

# List only the text classification models

api.list_models(filter="text-classification")

# Using the `ModelFilter`

filt = ModelFilter(task="text-classification")

# With `ModelSearchArguments`

filt = ModelFilter(task=args.pipeline_tags.TextClassification)

api.list_models(filter=filt)

# Using `ModelFilter` and `ModelSearchArguments` to find text classification in both PyTorch and TensorFlow

filt = ModelFilter(

    task=args.pipeline_tags.TextClassification,

    library=[args.library.PyTorch, args.library.TensorFlow],

)

api.list_models(filter=filt)

# List only models from the AllenNLP library

api.list_models(filter="allennlp")

# Using `ModelFilter` and `ModelSearchArguments`

filt = ModelFilter(library=args.library.allennlp)

Example usage with the search argument:

from huggingface_hub import HfApi

api = HfApi()

# List all models with "bert" in their name

api.list_models(search="bert")

# List all models with "bert" in their name made by google

api.list_models(search="bert", author="google")

list_repo_files
< source >

( repo_id: strrevision: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = Nonetimeout: typing.Optional[float] = Nonetoken: typing.Union[bool, str, NoneType] = None ) → List[str]

Parameters

repo_id (str) — A namespace (user or an organization) and a repo name separated by a /.
revision (str, optional) — The revision of the model repository from which to get the information.
repo_type (str, optional) — Set to "dataset" or "space" if uploading to a dataset or space, None or "model" if uploading to a model. Default is None.
timeout (float, optional) — Whether to set a timeout for the request to the Hub.

    token (bool or str, optional) — A valid authentication token (see https://huggingface.co/settings/token). If None or True and machine is logged in (through huggingface-cli login or login()), token will be retrieved from the cache. If False, token is not sent in the request header. 

Returns

List[str]

the list of files in a given repository.

Get the list of files in a given repo.
list_spaces
< source >

( filter: typing.Union[str, typing.Iterable[str], NoneType] = Noneauthor: typing.Optional[str] = Nonesearch: typing.Optional[str] = Nonesort: typing.Union[typing.Literal['lastModified'], str, NoneType] = Nonedirection: typing.Optional[typing.Literal[-1]] = Nonelimit: typing.Optional[int] = Nonedatasets: typing.Union[str, typing.Iterable[str], NoneType] = Nonemodels: typing.Union[str, typing.Iterable[str], NoneType] = Nonelinked: bool = Falsefull: typing.Optional[bool] = Nonetoken: typing.Optional[str] = None ) → List[SpaceInfo]

Parameters

filter str or Iterable, optional) — A string tag or list of tags that can be used to identify Spaces on the Hub.
author (str, optional) — A string which identify the author of the returned Spaces.
search (str, optional) — A string that will be contained in the returned Spaces.
sort (Literal["lastModified"] or str, optional) — The key with which to sort the resulting Spaces. Possible values are the properties of the huggingface_hub.hf_api.SpaceInfo` class.
direction (Literal[-1] or int, optional) — Direction in which to sort. The value -1 sorts by descending order while all other values sort by ascending order.
limit (int, optional) — The limit on the number of Spaces fetched. Leaving this option to None fetches all Spaces.
datasets (str or Iterable, optional) — Whether to return Spaces that make use of a dataset. The name of a specific dataset can be passed as a string.
models (str or Iterable, optional) — Whether to return Spaces that make use of a model. The name of a specific model can be passed as a string.
linked (bool, optional) — Whether to return Spaces that make use of either a model or a dataset.
full (bool, optional) — Whether to fetch all Spaces data, including the lastModified and the cardData.

    token (bool or str, optional) — A valid authentication token (see https://huggingface.co/settings/token). If None or True and machine is logged in (through huggingface-cli login or login()), token will be retrieved from the cache. If False, token is not sent in the request header. 

Returns

List[SpaceInfo]

a list of huggingface_hub.hf_api.SpaceInfo objects. To anticipate future pagination, please consider the return value to be a simple iterator.

Get the public list of all Spaces on huggingface.co
merge_pull_request
< source >

( repo_id: strdiscussion_num: inttoken: typing.Optional[str] = Nonecomment: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = None ) → DiscussionStatusChange

Parameters

repo_id (str) — A namespace (user or an organization) and a repo name separated by a /.
discussion_num (int) — The number of the Discussion or Pull Request . Must be a strictly positive integer.
comment (str, optional) — An optional comment to post with the status change.
repo_type (str, optional) — Set to "dataset" or "space" if uploading to a dataset or space, None or "model" if uploading to a model. Default is None.

    token (str, optional) — An authentication token (See https://huggingface.co/settings/token) 

Returns

DiscussionStatusChange

the status change event

Merges a Pull Request.

Raises the following errors:

    HTTPError if the HuggingFace API returned an error
    ValueError if some parameter value is invalid
    RepositoryNotFoundError If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to private and you do not have access.

model_info
< source >

( repo_id: strrevision: typing.Optional[str] = Nonetimeout: typing.Optional[float] = NonesecurityStatus: typing.Optional[bool] = Nonefiles_metadata: bool = Falsetoken: typing.Union[bool, str, NoneType] = None ) → huggingface_hub.hf_api.ModelInfo

Parameters

repo_id (str) — A namespace (user or an organization) and a repo name separated by a /.
revision (str, optional) — The revision of the model repository from which to get the information.
timeout (float, optional) — Whether to set a timeout for the request to the Hub.
securityStatus (bool, optional) — Whether to retrieve the security status from the model repository as well.
files_metadata (bool, optional) — Whether or not to retrieve metadata for files in the repository (size, LFS metadata, etc). Defaults to False.

    token (bool or str, optional) — A valid authentication token (see https://huggingface.co/settings/token). If None or True and machine is logged in (through huggingface-cli login or login()), token will be retrieved from the cache. If False, token is not sent in the request header. 

Returns

huggingface_hub.hf_api.ModelInfo

The model repository information.

Get info on one specific model on huggingface.co

Model can be private if you pass an acceptable token or are logged in.

Raises the following errors:

    RepositoryNotFoundError If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to private and you do not have access.
    RevisionNotFoundError If the revision to download from cannot be found.

move_repo
< source >

( from_id: strto_id: strrepo_type: typing.Optional[str] = Nonetoken: typing.Optional[str] = None )

Parameters

from_id (str) — A namespace (user or an organization) and a repo name separated by a /. Original repository identifier.
to_id (str) — A namespace (user or an organization) and a repo name separated by a /. Final repository identifier.
repo_type (str, optional) — Set to "dataset" or "space" if uploading to a dataset or space, None or "model" if uploading to a model. Default is None.

    token (str, optional) — An authentication token (See https://huggingface.co/settings/token) 

Moving a repository from namespace1/repo_name1 to namespace2/repo_name2

Note there are certain limitations. For more information about moving repositories, please see https://hf.co/docs/hub/main#how-can-i-rename-or-transfer-a-repo.

Raises the following errors:

    RepositoryNotFoundError If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to private and you do not have access.

rename_discussion
< source >

( repo_id: strdiscussion_num: intnew_title: strtoken: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = None ) → DiscussionTitleChange

Parameters

repo_id (str) — A namespace (user or an organization) and a repo name separated by a /.
discussion_num (int) — The number of the Discussion or Pull Request . Must be a strictly positive integer.
new_title (str) — The new title for the discussion
repo_type (str, optional) — Set to "dataset" or "space" if uploading to a dataset or space, None or "model" if uploading to a model. Default is None.

    token (str, optional) — An authentication token (See https://huggingface.co/settings/token) 

Returns

DiscussionTitleChange

the title change event

Renames a Discussion.

Examples:

new_title = "New title, fixing a typo"

HfApi().rename_discussion(

    repo_id="username/repo_name",

    discussion_num=34

    new_title=new_title

)
# DiscussionTitleChange(id='deadbeef0000000', type='title-change', ...)

Raises the following errors:

    HTTPError if the HuggingFace API returned an error
    ValueError if some parameter value is invalid
    RepositoryNotFoundError If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to private and you do not have access.

repo_info
< source >

( repo_id: strrevision: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = Nonetimeout: typing.Optional[float] = Nonefiles_metadata: bool = Falsetoken: typing.Union[bool, str, NoneType] = None ) → Union[SpaceInfo, DatasetInfo, ModelInfo]

Parameters

repo_id (str) — A namespace (user or an organization) and a repo name separated by a /.
revision (str, optional) — The revision of the repository from which to get the information.
timeout (float, optional) — Whether to set a timeout for the request to the Hub.
files_metadata (bool, optional) — Whether or not to retrieve metadata for files in the repository (size, LFS metadata, etc). Defaults to False.

    token (bool or str, optional) — A valid authentication token (see https://huggingface.co/settings/token). If None or True and machine is logged in (through huggingface-cli login or login()), token will be retrieved from the cache. If False, token is not sent in the request header. 

Returns

Union[SpaceInfo, DatasetInfo, ModelInfo]

The repository information, as a huggingface_hub.hf_api.DatasetInfo, huggingface_hub.hf_api.ModelInfo or huggingface_hub.hf_api.SpaceInfo object.

Get the info object for a given repo of a given type.

Raises the following errors:

    RepositoryNotFoundError If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to private and you do not have access.
    RevisionNotFoundError If the revision to download from cannot be found.

set_access_token
< source >

( access_token: str )

Parameters

    access_token (str) — The access token to save. 

Saves the passed access token so git can correctly authenticate the user.
space_info
< source >

( repo_id: strrevision: typing.Optional[str] = Nonetimeout: typing.Optional[float] = Nonefiles_metadata: bool = Falsetoken: typing.Union[bool, str, NoneType] = None ) → SpaceInfo

Parameters

repo_id (str) — A namespace (user or an organization) and a repo name separated by a /.
revision (str, optional) — The revision of the space repository from which to get the information.
timeout (float, optional) — Whether to set a timeout for the request to the Hub.
files_metadata (bool, optional) — Whether or not to retrieve metadata for files in the repository (size, LFS metadata, etc). Defaults to False.

    token (bool or str, optional) — A valid authentication token (see https://huggingface.co/settings/token). If None or True and machine is logged in (through huggingface-cli login or login()), token will be retrieved from the cache. If False, token is not sent in the request header. 

Returns

SpaceInfo

The space repository information.

Get info on one specific Space on huggingface.co.

Space can be private if you pass an acceptable token.

Raises the following errors:

    RepositoryNotFoundError If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to private and you do not have access.
    RevisionNotFoundError If the revision to download from cannot be found.

unset_access_token
< source >

( )

Resets the user’s access token.
update_repo_visibility
< source >

( repo_id: strprivate: bool = Falsetoken: typing.Optional[str] = Noneorganization: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = Nonename: typing.Optional[str] = None )

Parameters

repo_id (str, optional) — A namespace (user or an organization) and a repo name separated by a /.
private (bool, optional, defaults to False) — Whether the model repo should be private.
token (str, optional) — An authentication token (See https://huggingface.co/settings/token)

    repo_type (str, optional) — Set to "dataset" or "space" if uploading to a dataset or space, None or "model" if uploading to a model. Default is None. 

Update the visibility setting of a repository.

Raises the following errors:

    RepositoryNotFoundError If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to private and you do not have access.

upload_file
< source >

( path_or_fileobj: typing.Union[str, bytes, typing.BinaryIO]path_in_repo: strrepo_id: strtoken: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = Nonerevision: typing.Optional[str] = Nonecommit_message: typing.Optional[str] = Nonecommit_description: typing.Optional[str] = Nonecreate_pr: typing.Optional[bool] = Noneparent_commit: typing.Optional[str] = None ) → str

Parameters

path_or_fileobj (str, bytes, or IO) — Path to a file on the local machine or binary data stream / fileobj / buffer.
path_in_repo (str) — Relative filepath in the repo, for example: "checkpoints/1fec34a/weights.bin"
repo_id (str) — The repository to which the file will be uploaded, for example: "username/custom_transformers"
token (str, optional) — Authentication token, obtained with HfApi.login method. Will default to the stored token.
repo_type (str, optional) — Set to "dataset" or "space" if uploading to a dataset or space, None or "model" if uploading to a model. Default is None.
revision (str, optional) — The git revision to commit from. Defaults to the head of the "main" branch.
commit_message (str, optional) — The summary / title / first line of the generated commit
commit_description (str optional) — The description of the generated commit
create_pr (boolean, optional) — Whether or not to create a Pull Request with that commit. Defaults to False. If revision is not set, PR is opened against the "main" branch. If revision is set and is a branch, PR is opened against this branch. If revision is set and is not a branch name (example: a commit oid), an RevisionNotFoundError is returned by the server.

    parent_commit (str, optional) — The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported. If specified and create_pr is False, the commit will fail if revision does not point to parent_commit. If specified and create_pr is True, the pull request will be created from parent_commit. Specifying parent_commit ensures the repo has not changed before committing the changes, and can be especially useful if the repo is updated / committed to concurrently. 

Returns

str

The URL to visualize the uploaded file on the hub

Upload a local file (up to 50 GB) to the given repo. The upload is done through a HTTP post request, and doesn’t require git or git-lfs to be installed.

Raises the following errors:

    HTTPError if the HuggingFace API returned an error
    ValueError if some parameter value is invalid
    RepositoryNotFoundError If the repository to download from cannot be found. This may be because it doesn’t exist, or because it is set to private and you do not have access.
    RevisionNotFoundError If the revision to download from cannot be found.

upload_file assumes that the repo already exists on the Hub. If you get a Client error 404, please make sure you are authenticated and that repo_id and repo_type are set correctly. If repo does not exist, create it first using create_repo().

Example:

from huggingface_hub import upload_file

with open("./local/filepath", "rb") as fobj:

    upload_file(

        path_or_fileobj=fileobj,

        path_in_repo="remote/file/path.h5",

        repo_id="username/my-dataset",

        repo_type="dataset",

        token="my_token",

    )
"https://huggingface.co/datasets/username/my-dataset/blob/main/remote/file/path.h5"

upload_file(

    path_or_fileobj=".\\local\\file\\path",

    path_in_repo="remote/file/path.h5",

    repo_id="username/my-model",

    token="my_token",

)
"https://huggingface.co/username/my-model/blob/main/remote/file/path.h5"

upload_file(

    path_or_fileobj=".\\local\\file\\path",

    path_in_repo="remote/file/path.h5",

    repo_id="username/my-model",

    token="my_token",

    create_pr=True,

)
"https://huggingface.co/username/my-model/blob/refs%2Fpr%2F1/remote/file/path.h5"

upload_folder
< source >

( repo_id: strfolder_path: typing.Union[str, pathlib.Path]path_in_repo: typing.Optional[str] = Nonecommit_message: typing.Optional[str] = Nonecommit_description: typing.Optional[str] = Nonetoken: typing.Optional[str] = Nonerepo_type: typing.Optional[str] = Nonerevision: typing.Optional[str] = Nonecreate_pr: typing.Optional[bool] = Noneparent_commit: typing.Optional[str] = Noneallow_patterns: typing.Union[typing.List[str], str, NoneType] = Noneignore_patterns: typing.Union[typing.List[str], str, NoneType] = None ) → str

Parameters

repo_id (str) — The repository to which the file will be uploaded, for example: "username/custom_transformers"
folder_path (str or Path) — Path to the folder to upload on the local file system
path_in_repo (str, optional) — Relative path of the directory in the repo, for example: "checkpoints/1fec34a/results". Will default to the root folder of the repository.
token (str, optional) — Authentication token, obtained with HfApi.login method. Will default to the stored token.
repo_type (str, optional) — Set to "dataset" or "space" if uploading to a dataset or space, None or "model" if uploading to a model. Default is None.
revision (str, optional) — The git revision to commit from. Defaults to the head of the "main" branch.
commit_message (str, optional) — The summary / title / first line of the generated commit. Defaults to: f"Upload {path_in_repo} with huggingface_hub"
commit_description (str optional) — The description of the generated commit
create_pr (boolean, optional) — Whether or not to create a Pull Request with that commit. Defaults to False. If revision is not set, PR is opened against the "main" branch. If revision is set and is a branch, PR is opened against this branch. If revision is set and is not a branch name (example: a commit oid), an RevisionNotFoundError is returned by the server.
parent_commit (str, optional) — The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported. If specified and create_pr is False, the commit will fail if revision does not point to parent_commit. If specified and create_pr is True, the pull request will be created from parent_commit. Specifying parent_commit ensures the repo has not changed before committing the changes, and can be especially useful if the repo is updated / committed to concurrently.
allow_patterns (List[str] or str, optional) — If provided, only files matching at least one pattern are uploaded.

    ignore_patterns (List[str] or str, optional) — If provided, files matching any of the patterns are not uploaded. 

Returns

str

A URL to visualize the uploaded folder on the hub

Upload a local folder to the given repo. The upload is done through a HTTP requests, and doesn’t require git or git-lfs to be installed.

The structure of the folder will be preserved. Files with the same name already present in the repository will be overwritten, others will be left untouched.

Use the allow_patterns and ignore_patterns arguments to specify which files to upload. These parameters accept either a single pattern or a list of patterns. Patterns are Standard Wildcards (globbing patterns) as documented here. If both allow_patterns and ignore_patterns are provided, both constraints apply. By default, all files from the folder are uploaded.

Uses HfApi.create_commit under the hood.

Raises the following errors:

    HTTPError if the HuggingFace API returned an error
    ValueError if some parameter value is invalid

upload_folder assumes that the repo already exists on the Hub. If you get a Client error 404, please make sure you are authenticated and that repo_id and repo_type are set correctly. If repo does not exist, create it first using create_repo().

Example:

upload_folder(

    folder_path="local/checkpoints",

    path_in_repo="remote/experiment/checkpoints",

    repo_id="username/my-dataset",

    repo_type="datasets",

    token="my_token",

    ignore_patterns="**/logs/*.txt",

)
# "https://huggingface.co/datasets/username/my-dataset/tree/main/remote/experiment/checkpoints"

upload_folder(

    folder_path="local/checkpoints",

    path_in_repo="remote/experiment/checkpoints",

    repo_id="username/my-dataset",

    repo_type="datasets",

    token="my_token",

    create_pr=True,

)
# "https://huggingface.co/datasets/username/my-dataset/tree/refs%2Fpr%2F1/remote/experiment/checkpoints"

whoami
< source >

( token: typing.Optional[str] = None )

Parameters

    token (str, optional) — Hugging Face token. Will default to the locally saved token if not provided. 

Call HF API to know “whoami”.
ModelInfo
class huggingface_hub.hf_api.ModelInfo
< source >

( modelId: typing.Optional[str] = Nonesha: typing.Optional[str] = NonelastModified: typing.Optional[str] = Nonetags: typing.Optional[typing.List[str]] = Nonepipeline_tag: typing.Optional[str] = Nonesiblings: typing.Optional[typing.List[typing.Dict]] = Noneprivate: bool = Falseauthor: typing.Optional[str] = Noneconfig: typing.Optional[typing.Dict] = NonesecurityStatus: typing.Optional[typing.Dict] = None**kwargs )

Parameters

modelId (str, optional) — ID of model repository.
sha (str, optional) — repo sha at this particular revision
lastModified (str, optional) — date of last commit to repo
tags (List[str], optional) — List of tags.
pipeline_tag (str, optional) — Pipeline tag to identify the correct widget.
siblings (List[RepoFile], optional) — list of (huggingface_hub.hf_api.RepoFile) objects that constitute the model.
private (bool, optional, defaults to False) — is the repo private
author (str, optional) — repo author
config (Dict, optional) — Model configuration information
securityStatus (Dict, optional) — Security status of the model. Example: {"containsInfected": False}

    kwargs (Dict, optional) — Kwargs that will be become attributes of the class. 

Info about a model accessible from huggingface.co
DatasetInfo
class huggingface_hub.hf_api.DatasetInfo
< source >

( id: typing.Optional[str] = Nonesha: typing.Optional[str] = NonelastModified: typing.Optional[str] = Nonetags: typing.Optional[typing.List[str]] = Nonesiblings: typing.Optional[typing.List[typing.Dict]] = Noneprivate: bool = Falseauthor: typing.Optional[str] = Nonedescription: typing.Optional[str] = Nonecitation: typing.Optional[str] = NonecardData: typing.Optional[dict] = None**kwargs )

Parameters

id (str, optional) — ID of dataset repository.
sha (str, optional) — repo sha at this particular revision
lastModified (str, optional) — date of last commit to repo
tags (Listr[str], optional) — List of tags.
siblings (List[RepoFile], optional) — list of huggingface_hub.hf_api.RepoFile objects that constitute the dataset.
private (bool, optional, defaults to False) — is the repo private
author (str, optional) — repo author
description (str, optional) — Description of the dataset
citation (str, optional) — Dataset citation
cardData (Dict, optional) — Metadata of the model card as a dictionary.

    kwargs (Dict, optional) — Kwargs that will be become attributes of the class. 

Info about a dataset accessible from huggingface.co
SpaceInfo
class huggingface_hub.hf_api.SpaceInfo
< source >

( id: typing.Optional[str] = Nonesha: typing.Optional[str] = NonelastModified: typing.Optional[str] = Nonesiblings: typing.Optional[typing.List[typing.Dict]] = Noneprivate: bool = Falseauthor: typing.Optional[str] = None**kwargs )

Parameters

id (str, optional) — id of space
sha (str, optional) — repo sha at this particular revision
lastModified (str, optional) — date of last commit to repo
siblings (List[RepoFile], optional) — list of huggingface_hub.hf_api.RepoFIle objects that constitute the Space
private (bool, optional, defaults to False) — is the repo private
author (str, optional) — repo author

    kwargs (Dict, optional) — Kwargs that will be become attributes of the class. 

Info about a Space accessible from huggingface.co

This is a “dataclass” like container that just sets on itself any attribute passed by the server.
RepoFile
class huggingface_hub.hf_api.RepoFile
< source >

( rfilename: strsize: typing.Optional[int] = NoneblobId: typing.Optional[str] = Nonelfs: typing.Optional[huggingface_hub.hf_api.BlobLfsInfo] = None**kwargs )

Parameters

rfilename (str) — file name, relative to the repo root. This is the only attribute that’s guaranteed to be here, but under certain conditions there can certain other stuff.
size (int, optional) — The file’s size, in bytes. This attribute is present when files_metadata argument of repo_info is set to True. It’s None otherwise.
blob_id (str, optional) — The file’s git OID. This attribute is present when files_metadata argument of repo_info is set to True. It’s None otherwise.

    lfs (BlobLfsInfo, optional) — The file’s LFS metadata. This attribute is present whenfiles_metadata argument of repo_info is set to True and the file is stored with Git LFS. It’s None otherwise. 

Data structure that represents a public file inside a repo, accessible from huggingface.co
CommitInfo
class huggingface_hub.CommitInfo
< source >

( commit_url: strcommit_message: strcommit_description: stroid: strpr_url: typing.Optional[str] = None )

Parameters

commit_url (str) — Url where to find the commit.
commit_message (str) — The summary (first line) of the commit that has been created.
commit_description (str) — Description of the commit that has been created. Can be empty.
oid (str) — Commit hash id. Example: "91c54ad1727ee830252e457677f467be0bfd8a57".
pr_url (str, optional) — Url to the PR that has been created, if any. Populated when create_pr=True is passed.
pr_revision (str, optional) — Revision of the PR that has been created, if any. Populated when create_pr=True is passed. Example: "refs/pr/1".

    pr_num (int, optional) — Number of the PR discussion that has been created, if any. Populated when create_pr=True is passed. Can be passed as discussion_num in get_discussion_details(). Example: 1. 

Data structure containing information about a newly created commit.

Returned by create_commit().
create_commit API

Below are the supported values for CommitOperation():
class huggingface_hub.CommitOperationAdd
< source >

( path_in_repo: strpath_or_fileobj: typing.Union[str, bytes, typing.BinaryIO] )

Parameters

path_in_repo (str) — Relative filepath in the repo, for example: "checkpoints/1fec34a/weights.bin"

    path_or_fileobj (str, bytes, or BinaryIO) — Either:
        a path to a local file (as str) to upload
        a buffer of bytes (bytes) holding the content of the file to upload
        a “file object” (subclass of io.BufferedIOBase), typically obtained with open(path, "rb"). It must support seek() and tell() methods.

Raises

ValueError

    ValueError — If path_or_fileobj is not one of str, bytes or io.BufferedIOBase.
    ValueError — If path_or_fileobj is a str but not a path to an existing file.
    ValueError — If path_or_fileobj is a io.BufferedIOBase but it doesn’t support both seek() and tell().

Data structure holding necessary info to upload a file to a repository on the Hub.
as_file
< source >

( )

A context manager that yields a file-like object allowing to read the underlying data behind path_or_fileobj.

Example:

operation = CommitOperationAdd(

       path_in_repo="remote/dir/weights.h5",

       path_or_fileobj="./local/weights.h5",

)
CommitOperationAdd(path_in_repo='remote/dir/weights.h5', path_or_fileobj='./local/weights.h5')

with operation.as_file() as file:

    content = file.read()

b64content
< source >

( )

The base64-encoded content of path_or_fileobj

Returns: bytes
class huggingface_hub.CommitOperationDelete
< source >

( path_in_repo: stris_folder: typing.Union[bool, typing.Literal['auto']] = 'auto' )

Parameters

path_in_repo (str) — Relative filepath in the repo, for example: "checkpoints/1fec34a/weights.bin" for a file or "checkpoints/1fec34a/" for a folder.

    is_folder (bool or Literal["auto"], optional) — Whether the Delete Operation applies to a folder or not. If “auto”, the path type (file or folder) is guessed automatically by looking if path ends with a ”/” (folder) or not (file). To explicitly set the path type, you can set is_folder=True or is_folder=False. 

Data structure holding necessary info to delete a file or a folder from a repository on the Hub.
Hugging Face local storage

huggingface_hub stores the authentication information locally so that it may be re-used in subsequent methods.

It does this using the HfFolder utility, which saves data at the root of the user.
class huggingface_hub.HfFolder
< source >

( )
delete_token
< source >

( )

Deletes the token from storage. Does not fail if token does not exist.
get_token
< source >

( ) → str or None

Returns

str or None

The token, None if it doesn’t exist.

Get token or None if not existent.

Note that a token can be also provided using the HUGGING_FACE_HUB_TOKEN environment variable.
save_token
< source >

( token: str )

Parameters

    token (str) — The token to save to the HfFolder 

Save token, creating folder as needed.
Filtering helpers

Some helpers to filter repositories on the Hub are available in the huggingface_hub package.
DatasetFilter
class huggingface_hub.DatasetFilter
< source >

( author: typing.Optional[str] = Nonebenchmark: typing.Union[typing.List[str], str, NoneType] = Nonedataset_name: typing.Optional[str] = Nonelanguage_creators: typing.Union[typing.List[str], str, NoneType] = Nonelanguage: typing.Union[typing.List[str], str, NoneType] = Nonemultilinguality: typing.Union[typing.List[str], str, NoneType] = Nonesize_categories: typing.Union[typing.List[str], str, NoneType] = Nonetask_categories: typing.Union[typing.List[str], str, NoneType] = Nonetask_ids: typing.Union[typing.List[str], str, NoneType] = None )

Parameters

author (str, optional) — A string or list of strings that can be used to identify datasets on the Hub by the original uploader (author or organization), such as facebook or huggingface.
benchmark (str or List, optional) — A string or list of strings that can be used to identify datasets on the Hub by their official benchmark.
dataset_name (str, optional) — A string or list of strings that can be used to identify datasets on the Hub by its name, such as SQAC or wikineural
language_creators (str or List, optional) — A string or list of strings that can be used to identify datasets on the Hub with how the data was curated, such as crowdsourced or machine_generated.
language (str or List, optional) — A string or list of strings representing a two-character language to filter datasets by on the Hub.
multilinguality (str or List, optional) — A string or list of strings representing a filter for datasets that contain multiple languages.
size_categories (str or List, optional) — A string or list of strings that can be used to identify datasets on the Hub by the size of the dataset such as 100K<n<1M or 1M<n<10M.
task_categories (str or List, optional) — A string or list of strings that can be used to identify datasets on the Hub by the designed task, such as audio_classification or named_entity_recognition.

    task_ids (str or List, optional) — A string or list of strings that can be used to identify datasets on the Hub by the specific task such as speech_emotion_recognition or paraphrase. 

A class that converts human-readable dataset search parameters into ones compatible with the REST API. For all parameters capitalization does not matter.

Examples:

from huggingface_hub import DatasetFilter

# Using author

new_filter = DatasetFilter(author="facebook")

# Using benchmark

new_filter = DatasetFilter(benchmark="raft")

# Using dataset_name

new_filter = DatasetFilter(dataset_name="wikineural")

# Using language_creator

new_filter = DatasetFilter(language_creator="crowdsourced")

# Using language

new_filter = DatasetFilter(language="en")

# Using multilinguality

new_filter = DatasetFilter(multilinguality="multilingual")

# Using size_categories

new_filter = DatasetFilter(size_categories="100K<n<1M")

# Using task_categories

new_filter = DatasetFilter(task_categories="audio_classification")

# Using task_ids

new_filter = DatasetFilter(task_ids="paraphrase")

ModelFilter
class huggingface_hub.ModelFilter
< source >

( author: typing.Optional[str] = Nonelibrary: typing.Union[typing.List[str], str, NoneType] = Nonelanguage: typing.Union[typing.List[str], str, NoneType] = Nonemodel_name: typing.Optional[str] = Nonetask: typing.Union[typing.List[str], str, NoneType] = Nonetrained_dataset: typing.Union[typing.List[str], str, NoneType] = Nonetags: typing.Union[typing.List[str], str, NoneType] = None )

Parameters

author (str, optional) — A string that can be used to identify models on the Hub by the original uploader (author or organization), such as facebook or huggingface.
library (str or List, optional) — A string or list of strings of foundational libraries models were originally trained from, such as pytorch, tensorflow, or allennlp.
language (str or List, optional) — A string or list of strings of languages, both by name and country code, such as “en” or “English”
model_name (str, optional) — A string that contain complete or partial names for models on the Hub, such as “bert” or “bert-base-cased”
task (str or List, optional) — A string or list of strings of tasks models were designed for, such as: “fill-mask” or “automatic-speech-recognition”
tags (str or List, optional) — A string tag or a list of tags to filter models on the Hub by, such as text-generation or spacy.

    trained_dataset (str or List, optional) — A string tag or a list of string tags of the trained dataset for a model on the Hub. 

A class that converts human-readable model search parameters into ones compatible with the REST API. For all parameters capitalization does not matter.

from huggingface_hub import ModelFilter

# For the author_or_organization

new_filter = ModelFilter(author_or_organization="facebook")

# For the library

new_filter = ModelFilter(library="pytorch")

# For the language

new_filter = ModelFilter(language="french")

# For the model_name

new_filter = ModelFilter(model_name="bert")

# For the task

new_filter = ModelFilter(task="text-classification")

# Retrieving tags using the `HfApi.get_model_tags` method

from huggingface_hub import HfApi

api = HfApi()
# To list model tags

api.get_model_tags()
# To list dataset tags

api.get_dataset_tags()

new_filter = ModelFilter(tags="benchmark:raft")

# Related to the dataset

new_filter = ModelFilter(trained_dataset="common_voice")

DatasetSearchArguments
class huggingface_hub.DatasetSearchArguments
< source >

( api: typing.Optional[ForwardRef('HfApi')] = None )

A nested namespace object holding all possible values for properties of datasets currently hosted in the Hub with tab-completion. If a value starts with a number, it will only exist in the dictionary

Example:

args = DatasetSearchArguments()

args.author_or_organization.huggingface

args.language.en

ModelSearchArguments
class huggingface_hub.ModelSearchArguments
< source >

( api: typing.Optional[ForwardRef('HfApi')] = None )

A nested namespace object holding all possible values for properties of models currently hosted in the Hub with tab-completion. If a value starts with a number, it will only exist in the dictionary

Example:

args = ModelSearchArguments()

args.author_or_organization.huggingface

args.language.en