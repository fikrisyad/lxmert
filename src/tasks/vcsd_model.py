import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

MAX_UTTERANCE_LENGTH = 25


class VCSDModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_UTTERANCE_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim

        # VCSD Classification head
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_classes)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, utterance, response, img):
        text = utterance + "<sep>" + response
        x = self.lxrt_encoder(text, img)
        logit = self.logit_fc(x)

        return logit