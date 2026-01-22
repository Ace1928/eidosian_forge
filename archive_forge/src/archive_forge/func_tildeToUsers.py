from twisted.web import resource
def tildeToUsers(request):
    if request.postpath and request.postpath[0][:1] == '~':
        request.postpath[:1] = ['users', request.postpath[0][1:]]
        request.path = '/' + '/'.join(request.prepath + request.postpath)