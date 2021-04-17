import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torchmetrics import F1

from src.param import args
from src.pretrain.qa_answer_table import load_lxmert_qa
from src.tasks.vcsd_model import VCSDModel
from src.tasks.vcsd_data import VCSDDataset, VCSDTorchDataset, VCSDEvaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_data_tuple(splits: str, bs: int, shuffle=False, drop_last=False) -> DataTuple:
    dset = VCSDDataset(splits)
    tset = VCSDTorchDataset(dset)
    evaluator = VCSDEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class VCSD:
    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=1024,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        # Model
        self.model = VCSDModel()

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        # if args.load_lxmert_qa is not None:
        #     load_lxmert_qa(args.load_lxmert_qa, self.model,
        #                    label2ans=self.train_tuple.dataset.label2ans)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)

        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            datumid2pred = {}
            for i, (datum_id, raw_image_id, image_id, utterance, response, img, target) in iter_wrapper(
                    enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                img, target = img.cuda(), target.cuda()
                logit = self.model(utterance, response, img)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for did, l in zip(datum_id, label.cpu().numpy()):
                    datumid2pred[did] = l
            tp, tn, fp, fn = evaluator.evaluate(datumid2pred)
            accu = (tp + tn) / (tp + fp + fn + tn)
            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, accu * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_accu, valid_prec, valid_rec, valid_f1 = self.evaluate(eval_tuple)
                if valid_accu > best_valid:
                    best_valid = valid_accu
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_accu * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        datumid2pred = {}
        for i, datum_tuple in enumerate(loader):
            datum_id, raw_image_id, image_id, utterance, response, img = datum_tuple[:6]  # avoid seeing ground truth
            with torch.no_grad():
                img = img.cuda()
                logit = self.model(utterance, response, img)
                score, label = logit.max(1)
                for did, l in zip(datum_id, label.cpu().numpy()):
                    datumid2pred[did] = l
        if dump is not None:
            evaluator.dump_result(datumid2pred, dump)
        return datumid2pred

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        # metric = F1(num_classes=2)
        #             average='macro',
        #             compute_on_step=False)

        datumid2preds = self.predict(eval_tuple, dump)
        # return eval_tuple.evaluator.evaluate(datumid2preds)
        tp, tn, fp, fn = eval_tuple.evaluator.evaluate(datumid2preds)

        accu = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fmeasure = 0
        if precision + recall > 0:
            fmeasure = 2 * precision * recall / (precision + recall)
        # F1 = metric.compu
        return accu, precision, recall, fmeasure

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        datumid2preds = {}
        for i, (datum_id, raw_image_id, image_id, utterance, response, img, target) in enumerate(loader):
            _, label = target.max(1)
            for did, l in zip(datum_id, label.cpu().numpy()):
                datumid2preds[i] = l
        return evaluator.evaluate(datumid2preds)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    vcsd = VCSD()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vcsd.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            vcsd.predict(
                get_data_tuple(args.test, bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'val' in args.test:
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vcsd.evaluate(
                get_data_tuple('val', bs=950,
                               shuffle=False, drop_last=False)
                # dump=os.path.join(args.output, 'minival_predict.json')
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', vcsd.train_tuple.dataset.splits)
        if vcsd.valid_tuple is not None:
            print('Splits in Valid data:', vcsd.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (vcsd.oracle_score(vcsd.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        vcsd.train(vcsd.train_tuple, vcsd.valid_tuple)
