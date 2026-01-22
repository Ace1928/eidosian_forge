import os
import json
import random
import tempfile
import subprocess
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
def gen_html(conversations, height, width, title, other_speaker, human_speaker, user_icon, alt_icon):
    """
    Generate HTML string for the given conversation.

    :param conversation:
        The conversation to be rendered (after pre-processing)
    :param height:
        Height of the HTML page
    :param width:
        Width of the HTML page
    :param title:
        Title of the HTML page
    :param other_speaker:
        The title of the model (grey boxes)
    :param human_speaker:
        Human speaker in the dialogs (blue boxes)

    :return: HTML string for the desired conversation
    """
    html_str = f'<html>\n<head>\n    <meta http-equiv="content-type" content="text/html; charset=utf-8">\n    <title> {title} </title>\n    <style type="text/css">\n        @media print{{\n            @page{{ margin: 0; size: {str(width)}in {str(height)}in; }}\n        }}\n        ul{{\n          list-style: none;\n        }}\n        .{other_speaker}_img_div{{\n          display: inline-block;\n          float: left;\n          margin: 18px 5px 0px -25px;\n        }}\n        .{human_speaker}_img_div{{\n          display: inline-block;\n          float: right;\n          margin: 18px 15px 5px 5px;\n        }}\n        .{other_speaker}_img{{\n            content:url({alt_icon});\n        }}\n        .{human_speaker}_img{{\n            content:url({user_icon});\n        }}\n        .{other_speaker}_p_div{{\n          float: left;\n        }}\n        .{human_speaker}_p_div{{\n          float:right;\n        }}\n        p{{\n          display:inline-block;\n          overflow-wrap: break-word;\n          border-radius: 30px;\n          padding: 10px 10px 10px 10px;\n          font-family: Helvetica, Arial, sans-serif;\n        }}\n        .clear{{\n            float: none;\n            clear: both;\n        }}\n        .{other_speaker}{{\n                background: #eee;\n                float: left;\n            }}\n        .{human_speaker}{{\n            background: #0084ff;\n            color: #fff;\n            float: right;\n        }}\n        .breaker{{\n            color: #bec3c9;\n            display: block;\n            height: 20px;\n            margin: 20px 20px 20px 20px;\n            text-align: center;\n            text-transform: uppercase;\n        }}\n        img{{\n          border-radius: 50px;\n          width: 50px;\n          height: 50px;\n        }}\n    </style>\n</head>\n<body>\n{gen_convo_ul(conversations)}\n</body>\n</html>\n    '
    return html_str