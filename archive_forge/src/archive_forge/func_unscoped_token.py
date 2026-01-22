import uuid
from keystoneauth1 import fixture
def unscoped_token():
    return fixture.V2Token(token_id='3e2813b7ba0b4006840c3825860b86ed', expires='2012-10-03T16:58:01Z', user_id='c4da488862bd435c9e6c0275a0d0e49a', user_name='exampleuser')