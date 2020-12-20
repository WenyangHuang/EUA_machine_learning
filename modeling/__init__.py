from .model import MetaModel


def build_model(in_dim=47, out_dim=4):
    return MetaModel(
        in_dim=in_dim,
        out_dim=out_dim,
    )
