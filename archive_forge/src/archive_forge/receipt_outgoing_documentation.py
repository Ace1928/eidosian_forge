import time
from yowsup.structs import ProtocolEntity, ProtocolTreeNode
from .receipt import ReceiptProtocolEntity

    delivered:
    If we send the following without "to" specified, whatsapp will consider the message delivered,
    but will not notify the sender.
    <receipt to="xxxxxxxxxxx@s.whatsapp.net" id="1415389947-15"></receipt>

    read
    <receipt to="xxxxxxxxxxx@s.whatsapp.net" id="1415389947-15" type="read"></receipt>

    multiple items:
    <receipt type="read" to="xxxxxxxxxxxx@s.whatsapp.net" id="1431364583-191">
        <list>
            <item id="1431364572-189"></item>
            <item id="1431364575-190"></item>
        </list>
    </receipt>
    