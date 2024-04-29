import torch

device = torch.device(torch.cuda.current_device())
#https://github.com/microsoft/DirectML/issues/414#issuecomment-1541319479
def sparse_to_dense(sparse_tensor):
    if device.type in ['dml', 'mps', 'xpu', 'privateuseone']:
        return sparse_tensor.to_dense()
    return sparse_tensor
