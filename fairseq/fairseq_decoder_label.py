# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch.nn as nn
from fairseq import utils
from torch import Tensor


class FairseqDecoderLabel(nn.Module):
    """Base class for decoders."""

    def __init__(self):
        super().__init__()
        # self.dictionary = dictionary
        self.onnx_trace = False

    def forward(self, encoder_out=None):
        """
        Args:
            encoder_out (dict, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the decoder's output of shape `(batch, labels)`
        """
        x = self.extract_features(
            encoder_out=encoder_out
        )
        return x

    def extract_features(self, encoder_out=None):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, labels)`
        """
        raise NotImplementedError

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "label" in sample
                label = sample["label"]
            else:
                label = None
            out = self.adaptive_softmax.get_log_prob(net_output[1], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[1]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)


    def prepare_for_onnx_export_(self):
        self.onnx_trace = True
