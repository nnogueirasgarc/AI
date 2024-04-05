# @STUDENT: do not change these {import}s
import torch
import numpy as np
# set random seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_num_threads(4)


def normalize(X, y):

    # TODO: normalize the data
    # (you can use the functions {torch.mean} and {torch.std})
    X_mean = torch.mean(X)
    y_mean = torch.mean(y)
    X_std = torch.std(X)
    y_std = torch.std(y)
    X = (X-X_mean)/X_std
    y = (y-y_mean)/y_std
    # return normalized data
    return [X, y]


def train_model(X, y):

    # TODO: choose appropriate values for the following parameters
    n_hidden_neurons = 1000
    learning_rate = 0.01
    n_epochs = 10000

    # TODO: make a neural network model as described in the instructions
    model = torch.nn.Sequential(torch.nn.Linear(1, n_hidden_neurons),torch.nn.Sigmoid(),torch.nn.Linear(n_hidden_neurons, 1))
    #model = torch.nn.Sequential(torch.nn.Linear(1, n_hidden_neurons),torch.nn.ReLU(),torch.nn.Linear(n_hidden_neurons, 1))

    
    # TODO: setup the loss function
    loss_fn = torch.nn.MSELoss(reduction='mean')

    # TODO: setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # RMSProp
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # RMSProp


    # print the loss every {n_epochs_print} epochs
    n_epochs_print = 50

    # TODO: uncomment the lines below to perform the training loop 
    #   and return the trained model
    
    # training loop
    for t in range(n_epochs):
        # TODO: implement the training loop
        y_pred = model(X)
        # TODO: uncomment the following lines to 
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print the loss
        if t % n_epochs_print == 0:
            print(f"Loss at epoch {t}: {loss.item()}")

    # all done
    return model
