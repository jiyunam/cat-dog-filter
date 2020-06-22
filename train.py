import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model import Net
from util import *
import pandas as pd
import time

def evaluate(net, loader, criterion):
    """ Evaluate the network on the validation set.

     Args:
         net: PyTorch neural network object
         loader: PyTorch data loader for the validation set
         criterion: The loss function
     Returns:
         err: A scalar for the average classification error over the validation set
         loss: A scalar for the average loss function over the validation set
     """
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0

    for i, data in enumerate(loader, 0):
        inputs, labels = data
        labels = normalize_label(labels)  # Convert labels to 0/1
        outputs = net(inputs)
        loss = criterion(outputs, labels.float())
        corr = (outputs > 0.0).squeeze().long() != labels
        total_err += int(corr.sum())
        total_loss += loss.item()
        total_epoch += len(labels)

    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)

    return err, loss

def main():
    ########################################################################
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(1000)

    ########################################################################
    # Loads the configuration for the experiment from the configuration file
    config, learning_rate, batch_size, num_epochs, target_classes = load_config('configuration.json')

    ########################################################################
    # Obtain the PyTorch data loader objects to load batches of the datasets
    train_loader, val_loader, test_loader, classes = get_data_loader(target_classes, batch_size)

    ########################################################################
    # Define a Convolutional Neural Network, defined in model.py
    net = Net()

    ########################################################################
    # Define the Loss function and optimizer
    # The loss function will be Binary Cross Entropy (BCE). In this case we
    # will use the BCEWithLogitsLoss which takes unnormalized output from
    # the neural network and scalar label.
    # Optimizer will be SGD with Momentum.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    ########################################################################
    # Set up some numpy arrays to store the training/test loss/erruracy
    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    ########################################################################
    # Train the network
    # Loop over the data iterator and sample a new batch of training data
    # Get the output from the network, and optimize our loss function.
    start_time = time.time()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        total_train_loss = 0.0
        total_train_err = 0.0

        total_epoch = 0

        for i, data in enumerate(train_loader, 0):
            # Get the inputs
            inputs, labels = data
            labels = normalize_label(labels) # Convert labels to 0/1

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass, backward pass, and optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # Calculate the statistics
            corr = (outputs > 0.0).squeeze().long() != labels
            total_train_err += int(corr.sum())
            total_train_loss += loss.item()
            total_epoch += len(labels)

        train_err[epoch] = float(total_train_err) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i+1)
        val_err[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion)

        print("Epoch {}: Train err: {}, Train loss: {} | Validation err: {}, Validation loss: {}".format(epoch + 1, train_err[epoch], train_loss[epoch], val_err[epoch], val_loss[epoch]))

    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    # Save the model to a file
    model_path = get_model_name(config)
    torch.save(net.state_dict(), model_path)

    # Write the train/test loss/err into CSV file for plotting later
    epochs = np.arange(1, num_epochs + 1)

    df = pd.DataFrame({"epoch": epochs, "train_err": train_err})
    df.to_csv("train_err_{}.csv".format(model_path), index=False)

    df = pd.DataFrame({"epoch": epochs, "train_loss": train_loss})
    df.to_csv("train_loss_{}.csv".format(model_path), index=False)

    df = pd.DataFrame({"epoch": epochs, "val_err": val_err})
    df.to_csv("val_err_{}.csv".format(model_path), index=False)

    df = pd.DataFrame({"epoch": epochs, "val_loss": val_loss})
    df.to_csv("val_loss_{}.csv".format(model_path), index=False)


if __name__ == '__main__':
    main()
