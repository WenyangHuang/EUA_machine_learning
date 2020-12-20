from typing import Any, Dict, List, Set

import torch.nn as nn
import torch.optim as optim


def build_optimizer(model, lr=1e-3):
    weight_decay = 1e-4

    params: List[Dict[str, Any]] = []
    memo: Set[nn.parameter.Parameter] = set()

    for module in model.modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            memo.add(value)
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = optim.SGD(params, lr=lr)
    
    return optimizer
