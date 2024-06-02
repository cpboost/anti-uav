# import torch
# import torch.utils.data.dataloader
# import importlib
# import collections
# from torch._six import string_classes
# from pytracking import TensorDict, TensorList
# int_classes = int

import jittor as jt
import jittor as jt
from jittor import Var
from jittor import contrib
import numpy as np
from collections import abc

def check_use_shared_memory():
    # 在 Jittor 中，我们可以通过检查 jt.flags.num_workers 来判断是否处于多进程环境中
    return jt.flags.num_workers > 0

# 使用示例
if check_use_shared_memory():
    print("Using shared memory for multi-process data loading.")
else:
    print("Not using shared memory for data loading.")





def ltr_collate_jittor(batch):
    """Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], jt.Var):
        # Jittor does not have a direct equivalent of _check_use_shared_memory,
        # so we will skip this part and assume that we are not in a background process.
        return jt.contrib.concat([b.unsqueeze(0) for b in batch], dim=0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np.issubdtype(elem.dtype, np.str_) or np.issubdtype(elem.dtype, np.object_):
                raise TypeError(error_msg.format(elem.dtype))

            return jt.contrib.concat([jt.Var(b) for b in batch], dim=0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return jt.Var(list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return jt.Var(batch).int()
    elif isinstance(batch[0], float):
        return jt.Var(batch).float()
    elif isinstance(batch[0], (str, bytes)):
        return batch
    elif isinstance(batch[0], dict):
        return {key: ltr_collate_jittor([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], (list, tuple)) and isinstance(batch[0][0], abc.Mapping):
        transposed = zip(*batch)
        return [ltr_collate_jittor(samples) for samples in transposed]
    elif isinstance(batch[0], (list, tuple)):
        transposed = zip(*batch)
        return jt.contrib.concat([ltr_collate_jittor(samples) for samples in transposed], dim=0)
    elif batch[0] is None:
        return batch

    raise TypeError((error_msg.format(type(batch[0]))))




import jittor as jt
from jittor.dataset import DataLoader

class JittorLTRLoader:
    def __init__(self, name, dataset, training=True, batch_size=1, shuffle=False, num_workers=0, epoch_interval=1, collate_fn=None, stack_dim=0):
        self.name = name
        self.training = training
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.epoch_interval = epoch_interval
        self.stack_dim = stack_dim

        # 设置 collate 函数
        if collate_fn is None:
            if stack_dim == 0:
                collate_fn = ltr_collate_jittor
            elif stack_dim == 1:
                # Jittor 没有直接等价于 PyTorch 的 ltr_collate_stack1
                # 你需要根据你的需求实现类似的逻辑
                raise NotImplementedError("Jittor does not have a direct equivalent of ltr_collate_stack1.")
            else:
                raise ValueError('Stack dim no supported. Must be 0 or 1.')

        # 创建数据集
        self.dataset = dataset

        # 创建数据加载器
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=collate_fn)

    def __iter__(self):
        return iter(self.dataloader)