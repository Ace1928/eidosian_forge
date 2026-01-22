import json
from urllib import request, parse
def web_search(query: str) -> List[str]:
    search_url = 'http://your-flask-server-url:5000/perform_search'
    data = json.dumps({'query': query}).encode()
    req = request.Request(search_url, data=data, headers={'content-type': 'application/json'})
    with request.urlopen(req) as response:
        result = json.loads(response.read().decode())
        return result