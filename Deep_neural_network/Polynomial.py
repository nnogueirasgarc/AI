# @STUDENT: do not change these {import}s
from libradry import training_loop
import torch
import numpy as np
# set random seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_num_threads(4)


class Cubic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # initialize a, b, c, d with random values
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        # TODO: edit the result return by this function appropriately
        return self.a * x **3 + self.b * x**2 + self.c*x + self.d


def train_model(X, y):

    # TODO: choose appropriate values for the following parameters
    learning_rate = 0.01
    n_epochs = 10000

    # TODO: create a cubic model as described in the instructions, 
    #   using the {Cubic} class implemented above
    model = Cubic()

    # TODO: setup the loss function
    loss_fn = torch.nn.MSELoss(reduction='mean')

    # TODO: setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # run the training loop
    training_loop(X, y, n_epochs, model, optimizer, loss_fn)

    # TODO: return the trained model
    # all done
    return model
