from .akld import akld
from .kld import kld
from metrics._warpper import batch_wise

metrics = ['akld', 'kld']
__all__ = metrics.copy()
for func_name in metrics:
    func = locals()[func_name]
    batch_func_name = f"batch_{func_name}"
    locals()[batch_func_name] = batch_wise(func)
    __all__.append(batch_func_name) 
