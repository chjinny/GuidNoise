import torch

def batch_wise(func):
    def wrapper(*args):
        for arg in args:
            assert type(arg) == torch.Tensor, 'The input should be 4D tensor'
            assert len(arg.shape) == 4, 'The input should be 4D tensor'
            assert arg.min() >= 0 and arg.max() <= 1, 'The value of tensor_a should be in [0, 1]'

        for i in range(args[0].shape[0]):
            yield func(*[a[i] for a in args])

    return wrapper