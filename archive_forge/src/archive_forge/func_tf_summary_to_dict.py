import io
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
import wandb.util
from wandb.sdk.lib import telemetry
from wandb.viz import custom_chart
def tf_summary_to_dict(tf_summary_str_or_pb: Any, namespace: str='') -> Optional[Dict[str, Any]]:
    """Convert a Tensorboard Summary to a dictionary.

    Accepts a tensorflow.summary.Summary, one encoded as a string,
    or a list of such encoded as strings.
    """
    values = {}
    if hasattr(tf_summary_str_or_pb, 'summary'):
        summary_pb = tf_summary_str_or_pb.summary
        values[namespaced_tag('global_step', namespace)] = tf_summary_str_or_pb.step
        values['_timestamp'] = tf_summary_str_or_pb.wall_time
    elif isinstance(tf_summary_str_or_pb, (str, bytes, bytearray)):
        summary_pb = Summary()
        summary_pb.ParseFromString(tf_summary_str_or_pb)
    elif hasattr(tf_summary_str_or_pb, '__iter__'):
        summary_pb = [Summary() for _ in range(len(tf_summary_str_or_pb))]
        for i, summary in enumerate(tf_summary_str_or_pb):
            summary_pb[i].ParseFromString(summary)
            if i > 0:
                summary_pb[0].MergeFrom(summary_pb[i])
        summary_pb = summary_pb[0]
    else:
        summary_pb = tf_summary_str_or_pb
    if not hasattr(summary_pb, 'value') or len(summary_pb.value) == 0:
        return None

    def encode_images(_img_strs: List[bytes], _value: Any) -> None:
        try:
            from PIL import Image
        except ImportError:
            wandb.termwarn('Install pillow if you are logging images with Tensorboard. To install, run `pip install pillow`.', repeat=False)
            return None
        if len(_img_strs) == 0:
            return None
        images: List[Union[wandb.Video, wandb.Image]] = []
        for _img_str in _img_strs:
            if _img_str.startswith(b'GIF'):
                images.append(wandb.Video(io.BytesIO(_img_str), format='gif'))
            else:
                images.append(wandb.Image(Image.open(io.BytesIO(_img_str))))
        tag_idx = _value.tag.rsplit('/', 1)
        if len(tag_idx) > 1 and tag_idx[1].isdigit():
            tag, idx = tag_idx
            values.setdefault(history_image_key(tag, namespace), []).extend(images)
        else:
            values[history_image_key(_value.tag, namespace)] = images
        return None
    for value in summary_pb.value:
        kind = value.WhichOneof('value')
        if kind in IGNORE_KINDS:
            continue
        if kind == 'simple_value':
            values[namespaced_tag(value.tag, namespace)] = value.simple_value
        elif kind == 'tensor':
            plugin_name = value.metadata.plugin_data.plugin_name
            if plugin_name == 'scalars' or plugin_name == '':
                values[namespaced_tag(value.tag, namespace)] = make_ndarray(value.tensor)
            elif plugin_name == 'images':
                img_strs = value.tensor.string_val[2:]
                encode_images(img_strs, value)
            elif plugin_name == 'histograms':
                ndarray = make_ndarray(value.tensor)
                if ndarray is None:
                    continue
                shape = ndarray.shape
                counts = []
                bins = []
                if shape[0] > 1:
                    bins.append(ndarray[0][0])
                    for v in ndarray:
                        counts.append(v[2])
                        bins.append(v[1])
                elif shape[0] == 1:
                    counts = [ndarray[0][2]]
                    bins = ndarray[0][:2]
                if len(counts) > 0:
                    try:
                        values[namespaced_tag(value.tag, namespace)] = wandb.Histogram(np_histogram=(counts, bins))
                    except ValueError:
                        wandb.termwarn('Not logging key "{}". Histograms must have fewer than {} bins'.format(namespaced_tag(value.tag, namespace), wandb.Histogram.MAX_LENGTH), repeat=False)
            elif plugin_name == 'pr_curves':
                pr_curve_data = make_ndarray(value.tensor)
                if pr_curve_data is None:
                    continue
                precision = pr_curve_data[-2, :].tolist()
                recall = pr_curve_data[-1, :].tolist()
                data = []
                for i in range(min(len(precision), len(recall))):
                    if precision[i] != 0 or recall[i] != 0:
                        data.append((recall[i], precision[i]))
                data = sorted(data, key=lambda x: (x[0], -x[1]))
                data_table = wandb.Table(data=data, columns=['recall', 'precision'])
                name = namespaced_tag(value.tag, namespace)
                values[name] = custom_chart('wandb/line/v0', data_table, {'x': 'recall', 'y': 'precision'}, {'title': f'{name} Precision v. Recall'})
        elif kind == 'image':
            img_str = value.image.encoded_image_string
            encode_images([img_str], value)
        elif kind == 'histo':
            tag = namespaced_tag(value.tag, namespace)
            if len(value.histo.bucket_limit) >= 3:
                first = value.histo.bucket_limit[0] + value.histo.bucket_limit[0] - value.histo.bucket_limit[1]
                last = value.histo.bucket_limit[-2] + value.histo.bucket_limit[-2] - value.histo.bucket_limit[-3]
                np_histogram = (list(value.histo.bucket), [first] + value.histo.bucket_limit[:-1] + [last])
                try:
                    values[tag] = wandb.Histogram(np_histogram=np_histogram)
                except ValueError:
                    wandb.termwarn(f'Not logging key {tag!r}. Histograms must have fewer than {wandb.Histogram.MAX_LENGTH} bins', repeat=False)
            else:
                wandb.termwarn(f'Not logging key {tag!r}. Found a histogram with only 2 bins.', repeat=False)
    return values