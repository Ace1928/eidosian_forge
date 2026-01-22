import argparse
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms as T
from transformers import (
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
def prepare_dummy_inputs(tokenizer, image_processor):
    prompt = 'Question answering. What is the name of the company?'
    prompt = 'Question answering. In which year is the report made?'
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    image = get_image()
    words = ['7', 'ITC', 'Limited', 'REPORT', 'AND', 'ACCOUNTS', '2013', 'ITC’s', 'Brands:', 'An', 'Asset', 'for', 'the', 'Nation', 'The', 'consumer', 'needs', 'and', 'aspirations', 'they', 'fulfil,', 'the', 'benefit', 'they', 'generate', 'for', 'millions', 'across', 'ITC’s', 'value', 'chains,', 'the', 'future-ready', 'capabilities', 'that', 'support', 'them,', 'and', 'the', 'value', 'that', 'they', 'create', 'for', 'the', 'country,', 'have', 'made', 'ITC’s', 'brands', 'national', 'assets,', 'adding', 'to', 'India’s', 'competitiveness.', 'It', 'is', 'ITC’s', 'aspiration', 'to', 'be', 'the', 'No', '1', 'FMCG', 'player', 'in', 'the', 'country,', 'driven', 'by', 'its', 'new', 'FMCG', 'businesses.', 'A', 'recent', 'Nielsen', 'report', 'has', 'highlighted', 'that', "ITC's", 'new', 'FMCG', 'businesses', 'are', 'the', 'fastest', 'growing', 'among', 'the', 'top', 'consumer', 'goods', 'companies', 'operating', 'in', 'India.', 'ITC', 'takes', 'justifiable', 'pride', 'that,', 'along', 'with', 'generating', 'economic', 'value,', 'these', 'celebrated', 'Indian', 'brands', 'also', 'drive', 'the', 'creation', 'of', 'larger', 'societal', 'capital', 'through', 'the', 'virtuous', 'cycle', 'of', 'sustainable', 'and', 'inclusive', 'growth.', 'DI', 'WILLS', '*', ';', 'LOVE', 'DELIGHTFULLY', 'SOFT', 'SKIN?', 'aia', 'Ans', 'Source:', 'https://www.industrydocuments.ucsf.edu/docs/snbx0223']
    boxes = [[0, 45, 67, 80], [72, 56, 109, 67], [116, 56, 189, 67], [198, 59, 253, 66], [257, 59, 285, 66], [289, 59, 365, 66], [372, 59, 407, 66], [74, 136, 161, 158], [175, 137, 306, 158], [318, 137, 363, 158], [374, 137, 472, 158], [483, 136, 529, 158], [540, 137, 593, 158], [608, 137, 717, 158], [73, 194, 100, 203], [106, 196, 177, 203], [183, 194, 227, 203], [233, 194, 259, 203], [265, 194, 344, 205], [74, 211, 104, 222], [109, 210, 141, 221], [147, 211, 169, 220], [175, 210, 223, 220], [229, 211, 259, 222], [265, 211, 329, 222], [334, 210, 352, 220], [74, 227, 127, 236], [133, 229, 180, 236], [187, 227, 221, 236], [226, 227, 264, 236], [270, 227, 320, 237], [327, 227, 349, 236], [74, 243, 161, 254], [166, 243, 249, 254], [254, 243, 281, 252], [286, 244, 342, 254], [74, 260, 112, 270], [119, 260, 145, 269], [151, 260, 174, 269], [179, 260, 217, 269], [222, 260, 249, 269], [254, 260, 285, 271], [290, 260, 335, 269], [340, 259, 359, 269], [74, 276, 95, 284], [101, 276, 156, 287], [164, 276, 198, 284], [203, 276, 244, 284], [251, 275, 285, 284], [291, 276, 340, 284], [74, 292, 129, 301], [135, 292, 185, 302], [192, 292, 242, 303], [248, 292, 261, 301], [267, 292, 312, 301], [74, 308, 195, 319], [75, 335, 82, 344], [88, 335, 98, 344], [105, 335, 138, 344], [144, 335, 214, 346], [220, 336, 233, 344], [239, 335, 256, 344], [262, 335, 283, 344], [290, 335, 309, 344], [316, 335, 320, 344], [74, 351, 119, 360], [126, 352, 170, 362], [176, 352, 186, 360], [192, 352, 214, 360], [220, 352, 276, 362], [282, 352, 326, 360], [333, 352, 349, 362], [74, 368, 89, 377], [95, 370, 124, 377], [129, 367, 175, 377], [181, 368, 266, 377], [272, 368, 283, 376], [289, 368, 333, 377], [74, 384, 126, 393], [134, 385, 175, 395], [181, 384, 206, 393], [212, 384, 292, 395], [298, 384, 325, 393], [330, 384, 366, 393], [74, 403, 103, 409], [109, 400, 154, 409], [161, 401, 241, 409], [247, 403, 269, 409], [275, 401, 296, 409], [302, 400, 349, 409], [74, 417, 131, 428], [137, 419, 186, 428], [192, 417, 214, 426], [219, 417, 242, 428], [248, 419, 319, 426], [74, 433, 119, 444], [125, 433, 204, 444], [210, 433, 278, 444], [285, 433, 295, 441], [302, 433, 340, 442], [75, 449, 98, 458], [104, 449, 142, 458], [146, 449, 215, 460], [221, 449, 258, 460], [263, 449, 293, 459], [300, 449, 339, 460], [74, 466, 101, 474], [108, 466, 185, 476], [191, 466, 261, 474], [267, 466, 309, 476], [315, 466, 354, 474], [74, 482, 151, 491], [158, 482, 201, 491], [208, 482, 258, 491], [263, 482, 292, 491], [298, 482, 333, 491], [338, 482, 360, 491], [74, 498, 131, 507], [137, 498, 150, 507], [156, 498, 197, 509], [202, 498, 257, 507], [263, 498, 310, 509], [74, 515, 128, 525], [134, 515, 156, 523], [161, 515, 218, 523], [223, 515, 261, 525], [267, 514, 280, 523], [74, 531, 156, 540], [162, 531, 188, 540], [195, 531, 257, 540], [263, 531, 315, 542], [871, 199, 878, 202], [883, 199, 908, 202], [894, 251, 904, 257], [841, 268, 841, 270], [784, 373, 811, 378], [816, 373, 896, 378], [784, 381, 811, 387], [815, 381, 847, 387], [645, 908, 670, 915], [692, 908, 712, 915], [220, 984, 285, 993], [293, 983, 779, 996]]
    text_list = []
    bbox_list = []
    for text, box in zip(words, boxes):
        if text == '':
            continue
        sub_tokens = tokenizer.tokenize(text)
        for sub_token in sub_tokens:
            text_list.append(sub_token)
            bbox_list.append(box)
    input_ids = tokenizer.convert_tokens_to_ids(text_list)
    input_ids = prompt_ids + input_ids
    bbox = [[0, 0, 0, 0]] * len(prompt_ids) + bbox_list
    pixel_values = image_processor(image, return_tensors='pt').pixel_values
    original_pixel_values = original_transform(image, image_size=image_processor.size['height']).unsqueeze(0)
    assert torch.allclose(original_pixel_values, pixel_values)
    print('Pixel values are ok!')
    return (torch.tensor(input_ids).unsqueeze(0), torch.tensor(bbox).unsqueeze(0).float(), pixel_values)