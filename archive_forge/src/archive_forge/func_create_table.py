from the local database to W&B in Tables format.
import wandb
from wandb.integration.prodigy import upload_dataset
import base64
import collections.abc
import io
import urllib
from copy import deepcopy
import pandas as pd
from PIL import Image
import wandb
from wandb import util
from wandb.plots.utils import test_missing
from wandb.sdk.lib import telemetry as wb_telemetry
def create_table(data):
    """Create a W&B Table.

    - Create/decode images from URL/Base64
    - Uses spacy to translate NER span data to visualizations.
    """
    table_df = pd.DataFrame(data)
    columns = list(table_df.columns)
    if 'spans' in table_df.columns and 'text' in table_df.columns:
        columns.append('spans_visual')
    if 'image' in columns:
        columns.append('image_visual')
    main_table = wandb.Table(columns=columns)
    matrix = table_df.to_dict(orient='records')
    en_core_web_md = util.get_module('en_core_web_md', required='part_of_speech requires `en_core_web_md` library, install with `python -m spacy download en_core_web_md`')
    nlp = en_core_web_md.load(disable=['ner'])
    for _i, document in enumerate(matrix):
        if 'spans_visual' in columns and 'text' in columns:
            document['spans_visual'] = None
            doc = nlp(document['text'])
            ents = []
            if 'spans' in document and document['spans'] is not None:
                for span in document['spans']:
                    if 'start' in span and 'end' in span and ('label' in span):
                        charspan = doc.char_span(span['start'], span['end'], span['label'])
                        ents.append(charspan)
                doc.ents = ents
                document['spans_visual'] = named_entity(docs=doc)
        if 'image' in columns:
            document['image_visual'] = None
            if 'image' in document and document['image'] is not None:
                isurl = urllib.parse.urlparse(document['image']).scheme in ('http', 'https')
                isbase64 = 'data:' in document['image'] and ';base64' in document['image']
                if isurl:
                    try:
                        im = Image.open(urllib.request.urlopen(document['image']))
                        document['image_visual'] = wandb.Image(im)
                    except urllib.error.URLError:
                        print('Warning: Image URL ' + str(document['image']) + ' is invalid.')
                        document['image_visual'] = None
                elif isbase64:
                    imgb64 = document['image'].split('base64,')[1]
                    try:
                        msg = base64.b64decode(imgb64)
                        buf = io.BytesIO(msg)
                        im = Image.open(buf)
                        document['image_visual'] = wandb.Image(im)
                    except base64.binascii.Error:
                        print('Warning: Base64 string ' + str(document['image']) + ' is invalid.')
                        document['image_visual'] = None
                else:
                    document['image_visual'] = wandb.Image(document['image'])
        values_list = list(document.values())
        main_table.add_data(*values_list)
    return main_table