from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
@classmethod
def from_bearer_token(cls, oauth2_bearer_token: str, twitter_users: Sequence[str], number_tweets: Optional[int]=100) -> TwitterTweetLoader:
    """Create a TwitterTweetLoader from OAuth2 bearer token."""
    tweepy = _dependable_tweepy_import()
    auth = tweepy.OAuth2BearerHandler(oauth2_bearer_token)
    return cls(auth_handler=auth, twitter_users=twitter_users, number_tweets=number_tweets)