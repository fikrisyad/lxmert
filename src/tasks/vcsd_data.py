import json
import os
import pickle

import numpy as np
import torch
import PIL as Image

from torch.utils.data import Dataset
from torchvision import transforms

from src.param import args
from src.utils import load_obj_tsv, load_csv

TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

VCSD_IMG_RAW = '/home/lr/fikrisyad/workspace/playground/image_conv/yunjey/data/resized_raw/'
VCSD_DATA_ROOT = '/home/lr/fikrisyad/workspace/playground/image_conv/yunjey/data/visgen_combined/'

SPLIT2NAME = {
    'train': 'train',
    'valid': 'val',
    'test': 'test',
}
CROP_SIZE = 224
IMG_TRANSFORM = transforms.Compose([
            # transforms.RandomCrop(args.crop_size),
            transforms.RandomCrop(CROP_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])


class VCSDDataset:
    """
    VCSD Dataset: format csv/tsv:
    row: raw_image_id, image_id, utterance, response, labels
    """

    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        idx = 0
        for split in self.splits:
            # self.data.extend(json.load(open("data/vqa/%s.json" % split)))
            for row in load_csv('{}static_{}_LIMIT-debug.csv'.format(VCSD_DATA_ROOT, split)):
                r = {
                    'id': idx,
                    'raw_image_id': row['raw_image_id'],
                    'image_id': row['image_id'],
                    'utterance': row['utterance'],
                    'label': row['labels'],
                }
                self.data.append(r)
                idx += 1

        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['id']: datum
            for datum in self.data
        }

        # # Answers
        # self.ans2label = json.load(open("data/vqa/trainval_ans2label.json"))
        # self.label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
        # assert len(self.ans2label) == len(self.label2ans)

    @property
    # def num_answers(self):
    #     return len(self.ans2label)
    def __len__(self):
        return len(self.data)


class VCSDTorchDataset(Dataset):
    def __init__(self, dataset: VCSDDataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        raw_img_data = {}
        counter = 0
        for split in dataset.splits:
            for datum in dataset.data:
                if topk is not None and counter < topk:
                    break
                raw_img_path = os.path.join(VCSD_IMG_RAW, '{}.jpg'.format(datum['raw_image_id']))
                img = Image.open(raw_img_path).convert('RGB')
                img = IMG_TRANSFORM(img)
                raw_img_data[datum['id']] = {
                    'img_feat': img
                }
                counter += 1

        self.data = []
        for datum in self.raw_dataset.data:
            if datum['id'] in raw_img_data:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]
        datum_id = datum['id']
        raw_image_id = datum['raw_image_id']
        image_id = datum['image_id']
        utterance = datum['utterance']
        response = datum['response']
        # label = datum['label']
        img = self.raw_img_data[datum['id']]['img']

        if 'label' in datum:
            label = int(datum['label'])
            target = torch.zeros(2)
            target[label] = 1

            return datum_id, raw_image_id, image_id, utterance, response, img, target
        else:
            return datum_id, raw_image_id, image_id, utterance, response


class VCSDEvaluator:
    def __init__(self, dataset: VCSDDataset):
        self.datasets = dataset

    def evaluate(self, datumid2pred: dict):
        tp, tn, fp, fn = 0, 0, 0 , 0
        for datumid, pred in datumid2pred.items():
            datum = self.dataset.id2datum[datumid]
            label = datum['label']

            if label == pred:
                if label == 0:
                    tn += 1
                else:
                    tp += 1
            elif label != pred:
                if label == 0:
                    fp += 1
                else:
                    fn += 1

        return tp, tn, fp, fn


