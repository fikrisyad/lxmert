import json
import os
import pickle

import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from os import path

from src.param import args
from src.utils import load_obj_tsv, load_csv, write_to_csv

TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

VCSD_IMG_RAW = '/home/lr/fikrisyad/workspace/playground/image_conv/yunjey/data/resized_raw/'
VCSD_IMG_OG_PATH1 = '/home/lr/fikrisyad/workspace/playground/image_conv/yunjey/data/visgen/VG_100K/'
VCSD_IMG_OG_PATH2 = '/home/lr/fikrisyad/workspace/playground/image_conv/yunjey/data/visgen/VG_100K_2/'
VCSD_DATA_ROOT = '/home/lr/fikrisyad/workspace/playground/image_conv/yunjey/data/visgen_combined/'
# VCSD_FILE_BASE = 'vcsd_img_prediction_'
# VCSD_FILE_BASE = 'vcsd_bbox_topic_'
VCSD_FILE_BASE = 'vcsd_bbox_vgregions_'

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
            for row in load_csv('{}{}{}.csv'.format(VCSD_DATA_ROOT, VCSD_FILE_BASE, split), delimiter='\t'):
                r = {
                    'id': idx,
                    'raw_image_id': row['raw_image_id'],
                    'image_id': row['image_id'],
                    'utterance': row['utterance'],
                    'response': row['response'],
                    'label': row['label'],
                    'person1': row['person1'],
                    'person2': row['person2'],
                    'topic': row['topic']
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
    def __init__(self, dataset: VCSDDataset, resize_img: bool):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        self.raw_img_data = {}
        self.speakers_boxes = {}
        counter = 0
        for split in dataset.splits:
            for datum in dataset.data:
                if topk is not None and counter == topk:
                    break
                if resize_img:
                    raw_img_path = os.path.join(VCSD_IMG_RAW, '{}.jpg'.format(datum['raw_image_id']))
                else:
                    if path.exists(VCSD_IMG_OG_PATH1 + datum['raw_image_id'] + '.jpg'):
                        raw_img_path = os.path.join(VCSD_IMG_OG_PATH1, '{}.jpg'.format(datum['raw_image_id']))
                    else:
                        raw_img_path = os.path.join(VCSD_IMG_OG_PATH2, '{}.jpg'.format(datum['raw_image_id']))
                img = Image.open(raw_img_path).convert('RGB')
                img = IMG_TRANSFORM(img)
                self.raw_img_data[datum['id']] = {
                    'img_feat': img
                }

                xmin1, ymin1, width1, height1 = datum['person1'].split(";")[:4]
                xmin2, ymin2, width2, height2 = datum['person2'].split(";")[:4]
                xmin3, ymin3, width3, height3 = datum['topic'].split(";")
                self.speakers_boxes[datum['id']] = {
                    'person1_bbox': {
                        'xmin': float(xmin1),
                        'ymin': float(ymin1),
                        'xmax': float(xmin1) + float(width1),
                        'ymax': float(ymin1) + float(height1)
                    },
                    'person2_bbox': {
                        'xmin': float(xmin2),
                        'ymin': float(ymin2),
                        'xmax': float(xmin2) + float(width2),
                        'ymax': float(ymin2) + float(height2)
                    },
                    'topic_bbox': {
                        'xmin': float(xmin3),
                        'ymin': float(ymin3),
                        'xmax': float(xmin3) + float(width3),
                        'ymax': float(ymin3) + float(height3)
                    }
                }
                counter += 1

        self.data = []
        for datum in self.raw_dataset.data:
            if datum['id'] in self.raw_img_data:
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
        img = self.raw_img_data[datum['id']]['img_feat']
        person1 = self.speakers_boxes[datum['id']]['person1_bbox']
        person2 = self.speakers_boxes[datum['id']]['person2_bbox']
        topic = self.speakers_boxes[datum['id']]['topic_bbox']
        person1_bbox = torch.tensor([person1['xmin'], person1['ymin'], person1['xmax'], person1['ymax']]).unsqueeze(0)
        person2_bbox = torch.tensor([person2['xmin'], person2['ymin'], person2['xmax'], person2['ymax']]).unsqueeze(0)
        topic_bbox = torch.tensor([topic['xmin'], topic['ymin'], topic['xmax'], topic['ymax']]).unsqueeze(0)
        bboxes = torch.cat([person1_bbox, person2_bbox, topic_bbox], dim=0)

        if 'label' in datum:
            label = int(datum['label'])
            target = torch.zeros(2)
            target[label] = 1

            return datum_id, raw_image_id, image_id, utterance, response, img, bboxes, target
        else:
            return datum_id, raw_image_id, image_id, utterance, response, img


class VCSDEvaluator:
    def __init__(self, dataset: VCSDDataset):
        self.dataset = dataset

    def evaluate(self, datumid2pred: dict):
        tp, tn, fp, fn = 0, 0, 0, 0
        for datumid, pred in datumid2pred.items():
            datum = self.dataset.id2datum[datumid]
            label = datum['label']
            pred = int(pred)
            label = int(label)

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

    def eval_predictions(self, datumid2pred: dict):
        tp, fp = 0, 0
        for datumid, pred in datumid2pred.items():
            datum = self.dataset.id2datum[datumid]
            label = datum['label']
            raw_id = datum['raw_image_id']
            pred_datum = self.dataset.id2datum[pred]
            pred_raw_id = pred_datum['raw_image_id']

            if raw_id == pred_raw_id:
                tp += 1
            else:
                fp += 1
        return tp, fp

    def dump_results(self, datumid2pred: dict, path):
        fieldnames = ['image_id', 'utterance', 'response', 'raw_image_id', 'pred_raw_id']
        rows = []
        for datumid, pred in datumid2pred.items():
            datum = self.dataset.id2datum[datumid]
            pred_datum = self.dataset.id2datum[pred]

            utterance = datum['utterance']
            response = datum['response']
            raw_id = datum['raw_image_id']
            iid = datum['image_id']
            pred_raw_id = pred_datum['raw_image_id']
            rows.append({
                'image_id': iid,
                'utterance': utterance,
                'response': response,
                'raw_image_id': raw_id,
                'pred_raw_id': pred_raw_id
            })

        write_to_csv(path, fieldnames, rows, delimiter='\t')

    def dump_output(self, datumid2pred: dict, path):
        fieldnames = ['raw_image_id', 'image_id', 'utterance', 'response', 'label', 'pred']
        rows = []
        for datumid, pred in datumid2pred.items():
            datum = self.dataset.id2datum[datumid]
            label = datum['label']
            pred = int(pred)
            label = int(label)

            row = {
                'raw_image_id': datum['raw_image_id'],
                'image_id': datum['image_id'],
                'utterance': datum['utterance'],
                'response': datum['response'],
                'label': label,
                'pred': pred
            }
            rows.append(row)
        write_to_csv(path, fieldnames, rows, delimiter='\t')


"""
Dataset classes for data with all  VG regions
"""


class VCSDDatasetVGRegions:
    """
    VCSD Dataset with all VG Regions: format csv/tsv:
    row: raw_image_id, image_id, utterance, response, label, regions
    """

    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        idx = 0
        for split in self.splits:
            # self.data.extend(json.load(open("data/vqa/%s.json" % split)))
            for row in load_csv('{}{}{}.csv'.format(VCSD_DATA_ROOT, VCSD_FILE_BASE, split), delimiter='\t'):
                r = {
                    'id': idx,
                    'raw_image_id': row['raw_image_id'],
                    'image_id': row['image_id'],
                    'utterance': row['utterance'],
                    'response': row['response'],
                    'label': row['label'],
                    'regions': row['regions']
                }
                self.data.append(r)
                idx += 1

        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['id']: datum
            for datum in self.data
        }

    @property
    # def num_answers(self):
    #     return len(self.ans2label)
    def __len__(self):
        return len(self.data)


class VCSDTorchDatasetVGRegions(Dataset):
    def __init__(self, dataset: VCSDDatasetVGRegions, resize_img: bool):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        self.raw_img_data = {}
        self.bboxes = {}
        region_lengths = []
        counter = 0
        for split in dataset.splits:
            for datum in dataset.data:
                if topk is not None and counter == topk:
                    break
                if resize_img:
                    raw_img_path = os.path.join(VCSD_IMG_RAW, '{}.jpg'.format(datum['raw_image_id']))
                else:
                    if path.exists(VCSD_IMG_OG_PATH1 + datum['raw_image_id'] + '.jpg'):
                        raw_img_path = os.path.join(VCSD_IMG_OG_PATH1, '{}.jpg'.format(datum['raw_image_id']))
                    else:
                        raw_img_path = os.path.join(VCSD_IMG_OG_PATH2, '{}.jpg'.format(datum['raw_image_id']))
                img = Image.open(raw_img_path).convert('RGB')
                img = IMG_TRANSFORM(img)
                self.raw_img_data[datum['id']] = {
                    'img_feat': img
                }

                regions = datum['regions'].split(';')
                region_lengths.append(len(regions))
                boxes = []
                for region in regions:
                    if len(region) > 0:
                        x1, y1, x2, y2 = (float(x) for x in region.split(','))
                        boxes.append({
                            'xmin': x1,
                            'ymin': y1,
                            'xmax': x2,
                            'ymax': y2
                        })
                self.bboxes[datum['id']] = boxes

                counter += 1
        self.max_region = max(region_lengths)
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['id'] in self.raw_img_data:
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
        img = self.raw_img_data[datum['id']]['img_feat']
        # max_region = self.max_region
        max_region = 90
        bboxes_seqs = torch.zeros(max_region, 4)
        # bboxes_tensors = []
        bboxes = self.bboxes[datum_id]
        for i, box in enumerate(bboxes):
            if i < max_region:
                bbox_tensor = torch.tensor([box['xmin'], box['ymin'], box['xmax'], box['ymax']])
                # bboxes_tensors.append(bbox_tensor)
                bboxes_seqs[i] = bbox_tensor
            else:
                break

        if 'label' in datum:
            label = int(datum['label'])
            target = torch.zeros(2)
            target[label] = 1

            return datum_id, raw_image_id, image_id, utterance, response, img, bboxes_seqs, target
        else:
            return datum_id, raw_image_id, image_id, utterance, response, img, bboxes_seqs


