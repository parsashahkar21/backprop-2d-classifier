import numpy as np
from function import *

class Tensor:
    def __init__(self, data, requires_grad: bool = True, ctx=None):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self.ctx = ctx
    
    def __repr__(self):
        data_str = np.array2string(self.data, precision=4, suppress_small=True, prefix=' ' * 8)
        grad_str = np.array2string(self.grad, precision=4, suppress_small=True, prefix=' ' * 8)
        
        return (f"Tensor(shape={self.data.shape}\n"
                f" data: {data_str}\n"
                f" grad: {grad_str}")
        
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, False)
        return Add.apply(self, other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __neg__(self):
        return self.__mul__(-1)
    
    def __sub__(self, other):
        return self.__add__(-other)
    
    def __rsub__(self, other):
        return (-self).__add__(other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, False)
        return Mul.apply(self, other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, False)
        return Pow.apply(self, other)
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, False)
        return MatMul.apply(self, other)
    
    def sum(self, axis=None, keepdims=False):
        return Sum.apply(self, axis, keepdims)
    
    def backward(self):
        # 1) must start from a scalar
        if self.data.size != 1:
            raise ValueError("backward() can only be called on a scalar tensor (e.g., a loss).")
        # initialize grad dL/dL = 1
        self.grad = np.ones_like(self.data, dtype=np.float32)

        # 2) build topological order
        topo = []
        visited = set()

        def build_topo(t):
            tid = id(t)
            if tid in visited:
                return
            visited.add(tid)
            if t.ctx is not None:
                for parent in t.ctx.inputs:
                    build_topo(parent)
            topo.append(t)

        build_topo(self)

        # 3) backprop through graph
        for t in reversed(topo):
            if t.ctx is None:
                continue
            grads = t.ctx.grad_fn.backward(t.ctx, t.grad)
            if not isinstance(grads, tuple):
                grads = (grads,)
            for parent, g in zip(t.ctx.inputs, grads):
                if g is None:
                    continue
                parent.grad = parent.grad + g