import torch
from mambapy.mamba import Mamba, MambaConfig

config = MambaConfig(d_model=19, n_layers=1, d_state = 64)
model = Mamba(config)

B, L, D = 45, 700, 19
x = torch.randn(B, L, D)
y = model(x)

# assert y.shape == x.shape

print(x.size())
print(y.size())