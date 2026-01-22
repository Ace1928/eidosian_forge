import os
from os.path import abspath, expanduser
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from PIL import Image
from .utils import download_and_extract_archive, download_file_from_google_drive, extract_archive, verify_str_arg
from .vision import VisionDataset
def parse_train_val_annotations_file(self) -> None:
    filename = 'wider_face_train_bbx_gt.txt' if self.split == 'train' else 'wider_face_val_bbx_gt.txt'
    filepath = os.path.join(self.root, 'wider_face_split', filename)
    with open(filepath) as f:
        lines = f.readlines()
        file_name_line, num_boxes_line, box_annotation_line = (True, False, False)
        num_boxes, box_counter = (0, 0)
        labels = []
        for line in lines:
            line = line.rstrip()
            if file_name_line:
                img_path = os.path.join(self.root, 'WIDER_' + self.split, 'images', line)
                img_path = abspath(expanduser(img_path))
                file_name_line = False
                num_boxes_line = True
            elif num_boxes_line:
                num_boxes = int(line)
                num_boxes_line = False
                box_annotation_line = True
            elif box_annotation_line:
                box_counter += 1
                line_split = line.split(' ')
                line_values = [int(x) for x in line_split]
                labels.append(line_values)
                if box_counter >= num_boxes:
                    box_annotation_line = False
                    file_name_line = True
                    labels_tensor = torch.tensor(labels)
                    self.img_info.append({'img_path': img_path, 'annotations': {'bbox': labels_tensor[:, 0:4].clone(), 'blur': labels_tensor[:, 4].clone(), 'expression': labels_tensor[:, 5].clone(), 'illumination': labels_tensor[:, 6].clone(), 'occlusion': labels_tensor[:, 7].clone(), 'pose': labels_tensor[:, 8].clone(), 'invalid': labels_tensor[:, 9].clone()}})
                    box_counter = 0
                    labels.clear()
            else:
                raise RuntimeError(f'Error parsing annotation file {filepath}')