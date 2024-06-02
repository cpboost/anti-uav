import jittor as jt

# class MLP(jt.nn.Module):
#     """ Very simple multi-layer perceptron (also called FFN)"""
#
#     def __init__(self, input_dim=128, hidden_dim=64, output_dim=128, num_layers=3):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = jt.nn.ModuleList(jt.nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
#
#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = jt.nn.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x


import jittor as jt

a=jt.var([1,2,3])