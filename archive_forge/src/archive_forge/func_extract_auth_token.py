import base64
import httpx
from lazyops.libs import lazyload
from typing import Optional, List, Dict, Any, Union, Iterable, Generator, AsyncGenerator
from .base import BaseModel
from .user_data import AZUserData
from .claims import APIKeyJWTClaims
@classmethod
def extract_auth_token(cls, *data: Iterable[Union['Headers', Dict[str, str]]], settings: Optional['AuthZeroSettings']=None) -> Optional[str]:
    """
        Extract the Auth Token from the Headers, Cookies, or Dict
        """
    settings = settings or cls.get_settings()
    for item in data:
        if (token := cls.get_auth_token(data=item, settings=settings)):
            return token
    return None