import logging
import select
import socketserver
def forward_tunnel(local_port, remote_host, remote_port, transport):

    class SubHander(Handler):
        chain_host = remote_host
        chain_port = remote_port
        ssh_transport = transport
    ForwardServer(('127.0.0.1', local_port), SubHander).serve_forever()