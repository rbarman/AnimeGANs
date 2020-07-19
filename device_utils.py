import torch
'''
Utils to move data to device
    - source -> https://jovian.ml/aakashns/04-feedforward-nn/v/16#C32
    - DeviceDataLoader is a nice way to move each batch from data loader to device
        without explicitly writing code to move data in the training loop
'''

def get_default_device():
    '''Pick cuda GPU by default if available '''
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)