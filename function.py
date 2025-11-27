import numpy as np

__all__ = ["Add", "Mul", "Pow", "Log", "Sum", "MatMul", "ReLU", "Sigmoid", "Softmax", "CrossEntropyWithSoftmax"]

class Context:
    def __init__(self, grad_fn, *inputs):
        self.grad_fn = grad_fn
        self.inputs = inputs
            
class Function:
    @staticmethod
    def forward(ctx, *args): ...
    
    @staticmethod
    def backward(ctx, upstream_grad): ...

    @classmethod
    def apply(cls, *args):
        from tensor import Tensor
        inputs = [a for a in args if isinstance(a, Tensor)]
        ctx = Context(cls, *inputs)
        
        ctx.needs_input_grad = tuple(t.requires_grad for t in inputs)
        requires_grad = any(ctx.needs_input_grad)
        
        output_data = cls.forward(ctx, *args)
        ctx = ctx if requires_grad else None
        
        output_tensor = Tensor(output_data, requires_grad, ctx)
        
        return output_tensor

def unbroadcast_to(a, shape):
    while len(a.shape) > len(shape):
        a = a.sum(axis=0)
    
    for i, dim in enumerate(shape):
        if dim == 1:
            a = a.sum(axis=i, keepdims=True)

    return a

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a = a
        ctx.b = b
        return a.data + b.data
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = unbroadcast_to(upstream_grad, ctx.a.data.shape) if ctx.needs_input_grad[0] else None
        grad_b = unbroadcast_to(upstream_grad, ctx.b.data.shape) if ctx.needs_input_grad[1] else None
        
        return (grad_a, grad_b)
        
class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a = a
        ctx.b = b
        return a.data * b.data
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = unbroadcast_to(ctx.b * upstream_grad, ctx.a.data.shape) if ctx.needs_input_grad[0] else None
        grad_b = unbroadcast_to(ctx.a * upstream_grad, ctx.b.data.shape) if ctx.needs_input_grad[1] else None
        
        return (grad_a, grad_b)

class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a = a
        ctx.b = b
        return a.data @ b.data
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = upstream_grad @ ctx.b.data.T if ctx.needs_input_grad[0] else None
        grad_b = ctx.a.data.T @ upstream_grad if ctx.needs_input_grad[1] else None
        
        return (grad_a, grad_b)
            
class Pow(Function):
    @staticmethod
    def forward(ctx, base, exp):
        ctx.base = base
        ctx.exp = exp
        ctx.output = np.pow(base.data, exp.data)
        return ctx.output
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_base = ctx.exp.data * np.pow(ctx.base.data, ctx.exp.data - 1) * upstream_grad if ctx.needs_input_grad[0] else None
        grad_exp = np.log(ctx.base.data) * ctx.output * upstream_grad if ctx.needs_input_grad[1] else None
        
        return (grad_base, grad_exp)

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.a = a
        return np.log(a.data)
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = 1/ctx.a.data * upstream_grad if ctx.needs_input_grad[0] else None
        return (grad_a,)
        
class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        ctx.a = a
        ctx.axis = axis
        ctx.keepdims = keepdims
        return np.sum(a.data, axis=axis, keepdims=keepdims)
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = np.broadcast_to(upstream_grad, ctx.a.data.shape) if ctx.needs_input_grad[0] else None
        return (grad_a,)

# --------- NEW IMPLEMENTATIONS BELOW ---------

class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.mask = (a.data > 0).astype(np.float32)
        return np.maximum(a.data, 0.0)
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = upstream_grad * ctx.mask if ctx.needs_input_grad[0] else None
        return (grad_a,)

class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        out = 1.0 / (1.0 + np.exp(-a.data))
        ctx.out = out
        return out
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_a = upstream_grad * (ctx.out * (1.0 - ctx.out)) if ctx.needs_input_grad[0] else None
        return (grad_a,)

class Softmax(Function):
    @staticmethod
    def forward(ctx, a, axis):
        ctx.axis = axis
        shifted = a.data - a.data.max(axis=axis, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps, axis=axis, keepdims=True)
        ctx.out = probs
        return probs
    
    @staticmethod
    def backward(ctx, upstream_grad):
        # grad = y * (g - sum(g*y))
        s = ctx.out
        dot = np.sum(upstream_grad * s, axis=ctx.axis, keepdims=True)
        grad_a = s * (upstream_grad - dot)
        return (grad_a,)

class CrossEntropyWithSoftmax(Function):
    @staticmethod
    def forward(ctx, y_pred, y_true):
        # stable softmax
        shifted = y_pred.data - y_pred.data.max(axis=1, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        batch = y_pred.data.shape[0]
        # clip for numerical safety in log
        logp = np.log(probs + 1e-12)
        loss = -np.sum(y_true.data * logp) / batch

        ctx.probs = probs
        ctx.y_true = y_true.data
        ctx.batch = batch
        return np.array(loss, dtype=np.float32)
    
    @staticmethod
    def backward(ctx, upstream_grad):
        grad_y_pred = (ctx.probs - ctx.y_true) / ctx.batch
        grad_y_pred = grad_y_pred * upstream_grad  # upstream is typically 1.0
        # y_true is labels; no grad
        return (grad_y_pred, None)