import base64
import json
import logging
import re
import uuid
from xml.dom import minidom
from IPython.display import HTML, display
from rdkit import Chem
from rdkit.Chem import Draw
from . import rdMolDraw2D
def setEnabled(shouldEnable=True, quiet=False):
    """ Enable interactive molecule rendering """

    def _wrapMsgIntoDiv(uuid, msg, quiet):
        return f'<div class="lm-Widget p-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output"id="{uuid}">{('' if quiet else msg)}</div>'
    global _enabled_div_uuid
    loadingMsg = 'Loading rdkit-structure-renderer.js...'
    failedToLoadMsg = 'Failed to load rdkit-structure-renderer.js:'
    renderingMsg = 'Interactive molecule rendering is {0:s} in this Jupyter Notebook.'
    renderingUnavailableMsg = renderingMsg.format('not available')
    renderingEnabledMsg = renderingMsg.format('enabled')
    renderingDisabledMsg = renderingMsg.format('disabled')
    if not shouldEnable:
        _enabled_div_uuid = None
        return display(HTML(_wrapMsgIntoDiv('', renderingDisabledMsg, quiet)))
    if _enabled_div_uuid:
        return display(HTML(_wrapMsgIntoDiv(_enabled_div_uuid, renderingEnabledMsg, quiet)))
    _enabled_div_uuid = str(uuid.uuid1())
    return display(HTML(_wrapMsgIntoDiv(_enabled_div_uuid, loadingMsg, quiet) + f"""<script type="module">\nconst jsLoader = document.getElementById('{_enabled_div_uuid}') || {{}};\nconst setError = (e, resolve) => {{\n  jsLoader.innerHTML = (\n    '{failedToLoadMsg}<br>' +\n    e.toString() + '<br>' +\n    '{renderingUnavailableMsg}'\n  );\n  resolve && resolve();\n}};\nif (window.rdkStrRnr) {{\n  jsLoader.innerHTML = '{renderingEnabledMsg}';\n}} else {{\n  window.rdkStrRnr = new Promise(resolve => {{\n    try {{\n      fetch('{rdkitStructureRendererJsUrl}').then(\n        r => r.text().then(\n          t => import(URL.createObjectURL(new Blob([t], {{ type: 'application/javascript' }}))).then(\n            ({{ default: Renderer }}) => {{\n              const res = Renderer.init('{minimalLibJsUrl}');\n              return res.then(\n                Renderer => {{\n                  jsLoader.innerHTML = '{renderingEnabledMsg}';\n                  resolve(Renderer);\n                }}\n              ).catch(\n                e => setError(e, resolve)\n              );\n            }}\n          ).catch(\n              e => setError(e, resolve)\n          )\n        ).catch(\n          e => setError(e, resolve)\n        )\n      ).catch(\n        e => setError(e, resolve)\n      );\n    }} catch(e) {{\n      setError(e, resolve);\n    }}\n  }});\n}}\n</script>"""))