from tensor import Tensor
from typing import List, Dict
import numpy as np

class SGD:
    """
    Stochastic Gradient Descent with optional momentum.

    Args:
        params: list of Tensor parameters (requires_grad=True)
        lr: learning rate
        momentum: 0.0 disables momentum; typical values 0.8–0.95
    """
    def __init__(self, params: List[Tensor], lr: float = 0.01, momentum: float = 0.0):
        if not isinstance(params, list) or not all(isinstance(p, Tensor) for p in params):
            raise TypeError("`params` must be a list of `Tensor` objects")
        self.params = params
        self.lr = float(lr)
        self.momentum = float(momentum)
        self._velocity: Dict[int, np.ndarray] = {}  # per-parameter velocity

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            if self.momentum > 0.0:
                vid = id(p)
                v = self._velocity.get(vid)
                if v is None:
                    v = np.zeros_like(p.data)
                # v = mu * v - lr * grad
                v = self.momentum * v - self.lr * p.grad
                # param += v
                p.data += v
                self._velocity[vid] = v
            else:
                # vanilla SGD
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.fill(0.0)


class Adam:
    """
    Adam optimizer (Kingma & Ba, 2015).

    Args:
        params: list of Tensor parameters (requires_grad=True)
        lr: learning rate (default 1e-3)
        betas: (beta1, beta2) for first and second moment decay
        eps: numerical stability term
    """
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        if not isinstance(params, list) or not all(isinstance(p, Tensor) for p in params):
            raise TypeError("`params` must be a list of `Tensor` objects")
        self.params = params
        self.lr = float(lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)

        # per-parameter state
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.t: int = 0  # global step

    def step(self):
        self.t += 1
        b1t = 1.0 - self.beta1 ** self.t
        b2t = 1.0 - self.beta2 ** self.t

        for p in self.params:
            if p.grad is None:
                continue

            pid = id(p)
            m = self.m.get(pid)
            v = self.v.get(pid)
            if m is None:
                m = np.zeros_like(p.data)
            if v is None:
                v = np.zeros_like(p.data)

            # first and second moments
            m = self.beta1 * m + (1.0 - self.beta1) * p.grad
            v = self.beta2 * v + (1.0 - self.beta2) * (p.grad * p.grad)

            # bias-corrected moments
            m_hat = m / b1t
            v_hat = v / b2t

            # parameter update
            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))

            # store back
            self.m[pid] = m
            self.v[pid] = v

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.fill(0.0)