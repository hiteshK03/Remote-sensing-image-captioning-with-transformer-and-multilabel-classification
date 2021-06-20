import torch
import torch.nn.functional as F

from model import modules

from fairseq.models import FairseqEncoder, BaseFairseqModel
from fairseq.models import register_model, register_model_architecture, transformer_multi


def create_padding_mask(src_tokens, src_lengths):
    padding_mask = torch.zeros(src_tokens.shape[:2],
                               dtype=torch.bool,
                               device=src_tokens.device)

    for i, src_length in enumerate(src_lengths):
        padding_mask[i, src_length:] = 1

    return padding_mask


class SimplisticCaptioningEncoder(FairseqEncoder):
    def __init__(self, args):
        super().__init__(dictionary=None)
        self.feature_projection = modules.FeatureProjection(args) \
            if not args.no_projection else None
        self.spatial_encoding = modules.SpatialEncoding(args) \
            if args.feature_spatial_encoding else None

    def forward(self, src_tokens, src_lengths, src_locations, **kwargs):
        x = src_tokens

        if self.feature_projection is not None:
            x = self.feature_projection(src_tokens)
        if self.spatial_encoding is not None:
            x += self.spatial_encoding(src_locations)

        # B x T x C -> T x B x C
        enc_out = x.transpose(0, 1)

        # compute padding mask
        enc_padding_mask = create_padding_mask(src_tokens, src_lengths)

        return transformer_multi.EncoderOut(encoder_out=enc_out,
                                      encoder_padding_mask=enc_padding_mask,
                                      encoder_embedding=None,
                                      encoder_states=None)

    def reorder_encoder_out(self, encoder_out, new_order):
        enc_out = encoder_out.encoder_out
        enc_padding_mask = encoder_out.encoder_padding_mask

        return transformer_multi.EncoderOut(encoder_out=enc_out.index_select(1, new_order),
                                      encoder_padding_mask=enc_padding_mask.index_select(0, new_order),
                                      encoder_embedding=None,
                                      encoder_states=None)


class TransformerCaptioningEncoder(transformer_multi.TransformerEncoder):
    def __init__(self, args):
        super().__init__(args, None, modules.FeatureProjection(args))
        self.spatial_encoding = modules.SpatialEncoding(args) \
            if args.feature_spatial_encoding else None

    def forward(self, src_tokens, src_lengths, src_locations, **kwargs):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)

        if self.spatial_encoding is not None:
            x += self.spatial_encoding(src_locations)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = create_padding_mask(src_tokens, src_lengths)

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.layer_norm:
            x = self.layer_norm(x)

        return transformer_multi.EncoderOut(encoder_out=x,
                                      encoder_padding_mask=encoder_padding_mask,
                                      encoder_embedding=None,
                                      encoder_states=None)


class CaptioningModel(BaseFairseqModel):
    @staticmethod
    def add_args(parser):
        transformer_multi.TransformerModel.add_args(parser)
        parser.add_argument('--features-dim', type=int, default=2048,
                            help='visual features dimension')
        parser.add_argument('--feature-spatial-encoding', default=False, action='store_true',
                            help='use feature spatial encoding')

    @classmethod
    def build_model(cls, args, task):
        transformer_multi.base_architecture(args)

        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = transformer_multi.DEFAULT_MAX_TARGET_POSITIONS

        captions_dict = task.target_dictionary

        encoder = cls.do_build_encoder(args)
        decoder_cap = cls.do_build_decoder_cap(args, captions_dict)
        decoder_label = cls.do_build_decoder_label(args)
        return cls.do_build_model(encoder, decoder_cap, decoder_label)

    @classmethod
    def do_build_model(cls, encoder, decoder_cap, decoder_label):
        raise NotImplementedError

    @classmethod
    def do_build_encoder(cls, args):
        raise NotImplementedError

    @classmethod
    def do_build_decoder_cap(cls, args, captions_dict):
        decoder_embedding = transformer_multi.Embedding(num_embeddings=len(captions_dict),
                                                  embedding_dim=args.decoder_embed_dim,
                                                  padding_idx=captions_dict.pad())
        return transformer_multi.TransformerDecoder1(args, captions_dict, decoder_embedding)

    @classmethod
    def do_build_decoder_label(cls, args):
        return transformer_multi.TransformerDecoder2(args)

    def __init__(self, encoder, decoder_cap, decoder_label):
        super().__init__()
        self.encoder = encoder
        self.decoder_cap = decoder_cap
        self.decoder_label = decoder_label

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_cap_out = self.decoder_cap(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        decoder_label_out = self.decoder_label(encoder_out=encoder_out)

        return decoder_cap_out, decoder_label_out

    def forward_decoder_cap(self, prev_output_tokens, **kwargs):
        return self.decoder_cap(prev_output_tokens, **kwargs)

    def forward_decoder_label(self, **kwargs):
        return self.decoder_label(**kwargs)

    def max_decoder_positions(self):
        return self.decoder_cap.max_positions()


@register_model('default-captioning-model')
class DefaultCaptioningModel(CaptioningModel):
    @classmethod
    def do_build_encoder(cls, args):
        return TransformerCaptioningEncoder(args)

    @classmethod
    def do_build_model(cls, encoder, decoder_cap, decoder_label):
        return DefaultCaptioningModel(encoder, decoder_cap, decoder_label)


@register_model('simplistic-captioning-model')
class SimplisticCaptioningModel(CaptioningModel):
    @staticmethod
    def add_args(parser):
        CaptioningModel.add_args(parser)
        parser.add_argument('--no-projection', default=False, action='store_true',
                            help='do not project visual features')

    @classmethod
    def do_build_encoder(cls, args):
        return SimplisticCaptioningEncoder(args)

    @classmethod
    def do_build_model(cls, encoder, decoder_cap, decoder_label):
        return SimplisticCaptioningModel(encoder, decoder)


@register_model_architecture('default-captioning-model', 'default-captioning-arch')
def default_captioning_arch(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 3)


@register_model_architecture('simplistic-captioning-model', 'simplistic-captioning-arch')
def simplistic_captioning_arch(args):
    if args.no_projection:
        args.encoder_embed_dim = args.features_dim
