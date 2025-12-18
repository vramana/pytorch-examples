import torch
from collections import defaultdict


class UpdateRatioTracker:
    def __init__(self, log_every=20):
        self.total_updates = 0
        self.log_every = log_every
        self.step = 0
        self.history = defaultdict(list)

    @torch.no_grad()
    def record(self, model, optimizer):
        self.step += 1
        if self.step % self.log_every != 0:
            return

        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g_norm = p.grad.norm().item()
            w_norm = p.data.norm().item()
            lr = optimizer.param_groups[0]['lr']
            ratio = (g_norm / (w_norm + 1e-12) * lr)
            
            self.history[name].append(ratio)

