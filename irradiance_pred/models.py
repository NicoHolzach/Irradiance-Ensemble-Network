"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
models.py defines the classes of the different models that we 
want to train. It also defines the relevant functions used to 
actually train the model and tune the hyperparameters.
These models can then be used to predict the irradiance levels
to compare their forecast errors.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# Import general modules.
import os
import functools
import operator
import itertools
import gc
import time
from datetime import datetime
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Import DL and and vector modules.
import torch
from torch.optim import Adam, SGD
import torch.nn as nn
import numpy as np
import pandas as pd

# Import our own modules.
import data
import preprocessing
import getperformance

# Ensure to enable CUDNN.
torch.backends.cudnn.benchmark=True

# Define the CNN model that takes skyimages (and image sequences) as an input.
class cnnModel(nn.Module):
    """
    Class that defines the layers and forward pass method
    of our CNN.
    """
    def __init__(self, input_dim, cnn_num, conv_filters,
                conv_kernels, conv_strides, conv_paddings,
                maxpool_kernels, maxpool_strides, maxpool_paddings, 
                dense_num, dense_dropout, dense_activation, 
                dense_hidden_size):
        """
        Initializes the cnnModel object.

        Parameters
        ----------
        input_dim : tuple
            Input dimension of the image (sequence) inputs,
            e.g. (1, 200, 200) for the static sky images.
        cnn_num : int
            Number of CNN layers that this model should have.
            Convolutional layers, pooling layers, etc. are counted as 1.
            Dense layers at the end of model are NOT counted.
        conv_filters : list
            List of integers containing the number of convolutional
            filters that each convolutional layer should have,
            e.g. [32, 16] for a 2-layer CNN with 32 and 16 filters
        conv_kernels : list
            List of integers containing the size that each of
            the convolutional filters within each layer should have,
            e.g. [3, 2] for a 3x3 filter in the first, and a
            2x2 filter in the second layer.
        conv_strides : list
            List of integers containing the size of the stride that each
            convolutional filters within each layer should use
            e.g. [1, 2] for a stride size of 1 in the first layer and a
            stride size of 2 in the second layer of the conv. filters.
        conv_paddings : list
            List of integers containing the size of the paddings
            that should surround the matrix before applying the conv. filters
            e.g. [1, 1] for a padding of size 1 in both conv. layers.
        maxpool_kernels : list
            List of integers containing the size that each of
            the maxpool layers should have,
            e.g. [2, 1] for a 2x2 maxpooling in the first layer, and a
            1x1 maxpooling in the second layer.
        maxpool_strides : list
            List of integers containing the size of the stride that each
            maxpool operation within each layer should use
            e.g. [1, 2] for a maxpool stride size of 1 in the first layer and a
            maxpool stride size of 2 in the second layer of the conv. filters. 
        maxpool_paddings : list
            List of integers containing the size of the paddings
            that should surround the matrix before applying maxpooling
            e.g. [1, 1] for a padding of size 1 in both maxpool layers.
        dense_num : int
            Number of dense layers that are chained at the end of
            the CNN layers. Last dense layer is connected to the output.
        dense_dropout : float
            Which dropout probability should be used in the added
            dropout layer for the dense layers.
        dense_activation : torch.nn object
            Which activation function to use in the dense layers.
        dense_hidden_size : int  
            Which number of hidden neurons should be used for the dense layers.
        """

        super(cnnModel, self).__init__()

        # Create a dictionary that selects the correct CNN layers
        # based on the input. Images need 2D filters whereas
        # images3d need 3D filters.
        if len(input_dim) == 3: #images are of dimension 3
            layers = {'conv':nn.Conv2d,
                        'batch':nn.BatchNorm2d,
                        'maxpool':nn.MaxPool2d}
        elif len(input_dim) == 4: #images3d are of dimension 4
            layers = {'conv':nn.Conv3d,
                        'batch':nn.BatchNorm3d,
                        'maxpool':nn.MaxPool3d}

        # Initialize CNN layers.
        cnn_layers = []

        # Fill the CNN layers based on datatype
        # and the specified parameters per layer.
        for i in range(cnn_num):

            # The first CNN layer is special as it takes
            # the image(s) as an input whereas subsequent layers
            # take the convolutional layer output as input.
            if i == 0:
                in_channels = 1
            else:
                in_channels = conv_filters[i-1]
            
            # Add a convolution, batch normalization, ReLU activation,
            # dropout, and maxpooling per layer.
            cnn_layers.extend([
                layers['conv'](
                        in_channels=in_channels,
                        out_channels=conv_filters[i],
                        kernel_size=conv_kernels[i],
                        stride=conv_strides[i],
                        padding=conv_paddings[i]
                                ),
                layers['batch'](conv_filters[i]),
                nn.ReLU(inplace=True),
                layers['maxpool'](
                        kernel_size=maxpool_kernels[i],
                        stride=maxpool_strides[i],
                        padding=maxpool_paddings[i]
                                    )
                            ])

        # Unpack these CNN layers and add them to the model.
        self.cnn_layers = nn.Sequential(*cnn_layers)

        # Calculate the number of inputs going into the dense layers.
        num_features_dense = functools.reduce(operator.mul, list(self.cnn_layers(torch.rand(1, *input_dim)).shape))

        # Prepare the dense layers based on the specified parameters:
        # If we specify just one dense layer,
        # it is directly connected to our predicted irradiance value.
        if dense_num == 1:
            dense_layers = [nn.Linear(num_features_dense, 1)]
        
        # If we specify two dense layers, we add one additional linear layer
        # that is then connected to our predicted irradiance value.
        elif dense_num == 2:
            dense_layers = [nn.Linear(num_features_dense, dense_hidden_size), dense_activation, nn.Dropout(p=dense_dropout, inplace=True),
                            nn.Linear(dense_hidden_size, 1)]
        
        # For every additional dense layer, i.e. for dense_num > 3,
        # we add dense_num - 2 hidden layers of the size hidden_size.
        else:
            dense_layers = [nn.Linear(num_features_dense, dense_hidden_size), dense_activation, nn.Dropout(p=dense_dropout, inplace=True),
                            *[l for i in range(dense_num-2) for l in (nn.Linear(dense_hidden_size, dense_hidden_size),
                                                                    dense_activation,
                                                                    nn.Dropout(p=dense_dropout, inplace=True))],
                            nn.Linear(dense_hidden_size, 1)]

        # Unpack these dense layers and add them to the model.
        self.dense_layers = nn.Sequential(*dense_layers)

    # Method to pass the input X forward through our CNN.
    def forward(self, X):
        X = self.cnn_layers(X)
        X = X.view(X.size(0), -1) #flatten the input
        X = self.dense_layers(X)
        return X

   # Method to return the layers created at initialization.
    def get_cnn(self):
        return self.cnn_layers


# Define the LSTM model that takes irradiance, weather,
# and combined sequences as input.
class lstmModel(nn.Module):
    """
    Class that defines the LSTM for irradiance,
    weather, and combined sequence data.
    """
    def __init__(self, input_dim, lstm_hidden_size, lstm_num_layers, lstm_dropout, 
                dense_num, dense_dropout, dense_activation, dense_hidden_size):

        """
        Initializes the lstmModel object. 

        Parameters
        ----------
        input_dim : tuple
            Input dimension of the sequence inputs,
            e.g. (25, 7) for weather sequences of length 25.
        lstm_hidden_size : int
            Number of neurons in the LSTM hidden layer(s).
        lstm_num_layers : int
            Number of hidden LSTM layers.
        lstm_dropout : float
            Dropout probability applied to each LSTM layer.
        dense_num : int
            Number of dense layers chained after the LSTM layer(s).
            The last dense layer is connected to the output.
        dense_dropout : float
            Which dropout probability should be used in the added
            dropout layer for the dense layers.
        dense_activation : torch.nn object
            Which activation function to use in the dense layers.
        dense_hidden_size : int  
            Which number of hidden neurons should be used for the dense layers.
        """

        super(lstmModel, self).__init__()

        # Create the LSTM layers.
        self.lstm_layers = nn.LSTM(
            input_size=input_dim[-1], 
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers, 
            batch_first=True, 
            dropout=lstm_dropout)

        # Save some parts of the architecture to the model itself
        self.num_layers = lstm_num_layers
        self.hidden_size = lstm_hidden_size

        # Prepare the dense layers based on the specified parameters:
        # If we specify just one dense layer,
        # it is directly connected to our predicted irradiance value.
        if dense_num == 1:
            dense_layers = [nn.Linear(lstm_hidden_size, 1)]
        
        # If we specify two dense layers, we add one additional linear layer
        # that is then connected to our predicted irradiance value.
        elif dense_num == 2:
            dense_layers = [nn.Linear(lstm_hidden_size, dense_hidden_size), dense_activation, nn.Dropout(p=dense_dropout, inplace=True),
                            nn.Linear(dense_hidden_size, 1)]
        
        # For every additional dense layer, i.e. for dense_num > 3,
        # we add dense_num - 2 hidden layers of the size hidden_size.
        else:
            dense_layers = [nn.Linear(lstm_hidden_size, dense_hidden_size), dense_activation, nn.Dropout(p=dense_dropout, inplace=True),
                            *[l for i in range(dense_num-2) for l in (nn.Linear(dense_hidden_size, dense_hidden_size),
                                                                    dense_activation,
                                                                    nn.Dropout(p=dense_dropout, inplace=True))],
                            nn.Linear(dense_hidden_size, 1)]

        # Unpack these dense layers and add them to the model.
        self.dense_layers = nn.Sequential(*dense_layers)

    # Method to pass the X input through our LSTM.
    def forward(self, X):

        # Initialize the hidden state for first input using zeros.
        h0 = torch.zeros(self.num_layers, 
                        X.size(0), 
                        self.hidden_size, 
                        device=X.device,
                        requires_grad=True)

        # Same for the cell state.
        c0 = torch.zeros(self.num_layers, 
                        X.size(0), 
                        self.hidden_size, 
                        device=X.device,
                        requires_grad=True)

        # Propagate the X input through the LSTM.
        out, (hn, cn) = self.lstm_layers(X, (h0, c0))

        # Reshape the output so that it can fit into the dense layer
        # using this order logic: (batch_size, seq_length, hidden_size).
        out = out[:, -1, :]

        # Feed it into the dense layer and returns its output.
        out = self.dense_layers(out)

        return out


# Define our persistence baseline model.
class Persistence():
    """
    Class that defines our baseline persistence model.
    Takes the irradiance data as input
    but then uses the current irradiance level
    as its prediction (for all forecasting horizons).
    """
    def __init__(self):
        pass

    def __call__(self, X):
        """
        Method to return current irradiance given by X input
        to be used as y prediction of model.
        """
        pred = torch.from_numpy(np.array([x[-1] for x in X]))
        return pred.view(pred.size(0), -1)


# Define a neural network class used for our ensemble model.
class NN(nn.Module):
    """
    Class that defines the NN model for ensemble data.
    """
    def __init__(self, input_dim, hidden_num, hidden_size,
                dropout_prob):
        """
        Initializes NN object.

        Parameters
        ----------
        input_dim : tuple
            Tuple of the input dimensions of the ensemble input.
        hidden_num : int
            How many hidden layers the model should have.
        hidden_size : int
            How many hidden neurons each hidden layer should have.
        dropout_prob : float
            The dropout probability of the dropout layer that is
            added before each hidden layer.
        """

        super(NN, self).__init__()

        # Add the linear, activation, and dropout layers depending
        # on the specified number of hidden layers.
        # Let's first add the first hidden layer that is connected to the input.
        dense_layers = [
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob, inplace=True)]

        # If we want more than one hidden layer, add them.
        if hidden_num > 1:
            extra_layers = [*[l for i in range(hidden_num-1) for l in (
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Dropout(p=dropout_prob, inplace=True))]]
            dense_layers.extend(extra_layers)
        
        # Connect the last hidden layer to the output layer.
        dense_layers.append(nn.Linear(hidden_size, 1))
    
        # Unpack these dense layers and add them to the model.
        self.dense_layers = nn.Sequential(*dense_layers)

    # Define the forward pass.
    def forward(self, X):
        X = self.dense_layers(X)
        return X


# Define the class used to optimize our model parameters 
# during training. 
class Optimization:
    """
    Class used to train the different models.
    Contains optimzation parameters, such as the loss function,
    and also steps used to train and evaluate model parameters
    from epoch to epoch and from mini-batch to mini-batch.
    """
    def __init__(self, model, loss_fn, optimizer, GPU):

        """
        Initializes Optimization object.

        Parameters
        ----------
        model : {cnnModel, lstmModel, NN}
            Input one of the three class objects we use to construct 
            our CNN, LSTM, or NN ensemble models.
        loss_fn : torch.nn.Module or RMSELoss()
            Loss function used to calculate forecast errors.
        optimizer : torch.optim
            Torch optimizer object containing the optimizer we want
            to use, e.g. torch.optim.Adam.
        GPU : bool
            - If True, trains the model on a GPU.
            - If False, trains the model on a CPU.
        """

        # Initialize model architecture, loss function, and more.
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.runtime_forward_training = []
        self.runtime_backprop_training = []
        self.runtime_evaluation = []

        # Ise GPU if available and if specified.
        if GPU == True:
            if torch.cuda.is_available():
                self.model = self.model.cuda(0)
                self.loss_fn = self.loss_fn.cuda(0)
    
    # Method that describes one train step across one mini-batch.
    def train_step(self, X, y, GPU):
        """
        Method that implements one training step consisting of
        predicting on the train set, calculating the train error,
        and updating the model weights and biases.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of training inputs of minibatch.
        y : torch.Tensor
            Tensor of training targets of minibatch.
        GPU : bool
            - If True, trains the model on a GPU.
            - If False, trains the model on a CPU.

        Returns
        -------
        loss.item : 
            Returns training loss calculated by the loss
            function self.loss_fn.

        """
        # Set model to train mode, make predictions, and save runtime.
        start_time = time.time()
        self.model.train()
        yhat = self.model(X)
        end_time = time.time()
        self.runtime_forward_training.append(end_time - start_time)

        # Compute loss and gradients and start runtime.
        start_time = time.time()
        loss = self.loss_fn(y.unsqueeze(1), yhat)
        loss.backward()

        # Update parameters, restart gradients at zero, and save runtime.
        self.optimizer.step()
        self.optimizer.zero_grad()
        end_time = time.time()
        self.runtime_backprop_training.append(end_time - start_time)

        # Clean memory.
        del yhat
        gc.collect()
        if GPU == True:
            with torch.cuda.device('cuda:0'):
                torch.cuda.empty_cache()
        
        # Output loss.
        return loss.item()

    # Method that conducts multiple training step across mini-batches
    # to conduct one full-epoch of training steps.
    def train(self, train_loader, val_loader, filepath, n_epochs, 
                GPU, early_stopping, from_scratch, previous_losses):
        """
        Method that trains the model using mini-batches.

        Parameters
        ----------
        train_loader : torch.Dataloder
            Dataloader containing training data.
            This uses the output of the create_dataloader()
            function in preprocessing.py
        val_loader : torch.Dataloder
            Dataloader containing validation data.
            This uses the output of the create_dataloader()
            function in preprocessing.py
        filepath : str
            Path containing architecture, input type, and more to indicate
            where parameters and losses should be saved.
        n_epochs : int
            Number of epochs used for training.
        GPU : bool
            - If True, trains the model on a GPU.
            - If False, trains the model on a CPU.
        early_stopping : bool
            - If True, stops the training process before n_epochs has been reached
            if the validation loss has not improved for 10 epochs AND if
            a minimum of 20 epochs have passed
            - If False, stops the training process after n_epochs.
        from_scratch : bool
            - If True, trains the model from scratch.
            - If False, searches for trained model parameters with the same layer architecture,
            the same input, the same target variable, and the same hyperparameters. If they
            are available, continues training with these trained weigths and biases.
        previous_losses : None or pd.DataFrame
            - If None, effectively starts training the model from scratch.
            - If pd.DataFrame, prints the previous train and val losses and starts
            training from there.

        Returns
        -------
        None
            Only creates a .pt file in the ./parameters folder 
            containing the model parameters.
        """

        # Check whether from_scratch and previous_losses are used together correctly.
        if from_scratch == False and previous_losses is None:
            raise ValueError('If from_scratch=False, previous_losses must take a pd.DataFrame.')

        # Extract the previous losses and add them to the optimizer.
        if from_scratch == False:
            self.train_losses = list(previous_losses['train'].to_numpy())
            self.val_losses = list(previous_losses['val'].to_numpy())
            previous_epoch = previous_losses.index.max() + 1
            print('Starting with training of this model from previous checkpoint...')
        else:
            previous_epoch = 0
            print('Starting with training of this model from scratch...')
            
        # Set (and create) directory and path where model shall be saved.
        earlystopping_path = f'./parameters{filepath}__early_stopping(True).pt'
        param_path = f'./parameters{filepath}__early_stopping(False).pt'
        param_dir = os.path.dirname(param_path)
        if not os.path.exists(param_dir):
            os.makedirs(param_dir)
        
        # Train for specified number of epochs.
        for epoch in range(1, n_epochs+1):

            # If we are not training from scratch, only print the losses
            # up until the last epoch of the checkpoint from previous training.
            if from_scratch == False and epoch <= previous_epoch:
                
                # Print train and validation loss of previous training.
                if (epoch <= 20) | (epoch % 10 == 0):
                    print(f'[{epoch}/{n_epochs}] Train loss: {self.train_losses[epoch-1]:.4f}\t '
                            f'Val loss: {self.val_losses[epoch-1]:.4f}')
                continue

            # If we are training from scratch or have passed the previous number
            # of epochs, then actually train the model.
            else:
                    
                # Train the model 
                batch_losses = []

                # Iterate through minibatches and compute loss for each.
                for x_batch, y_batch in train_loader:

                    # Calculate minibatch loss, update models params, and add loss to list.
                    loss = self.train_step(x_batch, y_batch, GPU)
                    batch_losses.append(loss)

                # The training loss is the mean of all minibatch losses.
                training_loss = np.mean(batch_losses)
                self.train_losses.append(training_loss)

                # Clean memory.
                del loss
                gc.collect()
                if GPU == True:
                    with torch.cuda.device('cuda:0'):
                        torch.cuda.empty_cache()

                # Compute validation loss, hence we can disable gradient calulation.
                with torch.no_grad():
                    batch_val_losses = []

                    # Iterate through data loader and calculate val loss for each minibatch.
                    for x_val, y_val in val_loader:

                        # Start runtime.
                        start_time = time.time()

                        # Set model to evaluation model, make predictin, and compute minibatch loss.
                        self.model.eval()
                        yhat = self.model(x_val)
                        val_loss = self.loss_fn(y_val.unsqueeze(1), yhat).item()
                        batch_val_losses.append(val_loss)

                        # Save runtime.
                        end_time = time.time()
                        self.runtime_evaluation.append(end_time - start_time)

                    # Add loss to list; to be used for plotting.
                    validation_loss = np.mean(batch_val_losses)
                    self.val_losses.append(validation_loss)

                    # Clean memory.
                    del val_loss, yhat
                    gc.collect()
                    if GPU == True:
                        with torch.cuda.device('cuda:0'):
                            torch.cuda.empty_cache()

                # Save the train and val losses after each training epoch.
                loss_path = f'./performance{filepath}.pkl'
                loss_dir = os.path.dirname(loss_path)
                if not os.path.exists(loss_dir):
                        os.makedirs(loss_dir)
                loss_df = pd.DataFrame(
                    {'train':self.train_losses, 
                    'val':self.val_losses})
                loss_df.to_pickle(loss_path)

                # Clean the memory.
                del loss_path, loss_dir, loss_df
                gc.collect()
            
                # Save checkpoint of model and optimizer if the validation 
                # loss improved.
                best_loss = min(self.val_losses)
                if self.val_losses[-1] <= best_loss:

                    state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
                    torch.save(state, param_path)

                # Check whether the validation loss has improved in the last 10 epochs.
                if early_stopping == True and epoch > 20:

                    # Check if the last 10 validation losses were better 
                    # than that best loss.
                    stop_check = [i<=best_loss for i in self.val_losses[-10:]]

                    # If the validation loss has not improved in the last 10 epochs,
                    # terminate the training and rename the parameter file to 
                    # indicate that early stopping has occured.
                    if sum(stop_check) == 0:

                        # Rename parameter file.
                        os.rename(param_path, earlystopping_path)

                        # Terminate training.
                        print('The validation loss has not improved for 10 epochs.')
                        print('Terminating training for this model-input-horizon-hyperparam combination.')
                        break
                
                # Print progress during training.
                if (epoch <= 20) | (epoch % 10 == 0):
                    print(f'[{epoch}/{n_epochs}] Train loss: {self.train_losses[epoch-1]:.4f}\t '
                            f'Val loss: {self.val_losses[epoch-1]:.4f}')


# Define function to train multiple models in a row. 
# This is THE MAIN training function of this paper.
def train_multiple_models(modeltype, datatype, sample, sample_size, GPU, 
                        n_epochs, layers_list, delta_list, length_list, 
                        horizon_list, hyperparameter_dict, loss_fn, layers_only,
                        early_stopping, from_scratch=False):
    """
    Function to apply the Optimization() class to multiple model architectures,
    data inputs, forecast horizons, and hyperparameters in a row.

    Parameters
    ----------
    modeltype : {'CNN', 'LSTM'}
        Which type of model to train multiple versions of.
    datatype : {'images', 'images3d', 'irradiance', 'weather', 'combined'}
        Which data type to use to train the model
    sample : bool 
        Whether to use a sample of the data for training.
    sample_size: float or str
        - If float, gets the sample of the data according to a fixed fraction,
            of the dataset size, e.g. 0.1 of all data
        - If str, gets the sample according to a certain frequency between
		timestamps, e.g. '3min' for 3 minute deltas.
		See https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
		for all accepted frequences.
    GPU : bool
        - If True, trains the model on a GPU.
        - If False, trains the model on a CPU.
    n_epochs : int
        For how many epochs to train the model.
    layers_list : list 
        Array object including different model layer architectures.
    delta_list : list
        Array object including different deltas of the input data.
    length_list : list 
        Array object including different lengths of the input data.
    horizon_list : list
        Array object including different forecasting horizons of the target.
    hyperparameter_dict : dict
        Dictionary containing lists for each hyperparameter in this format:
        {'batch_size':[], 'lr':[], 'weight_decay':[]}
    loss: torch.nn obj
        Loss function to use for training.
    layers_only : bool
        Whether this function is used to ONLY train different layer architectures. 
        If True, this helps to speeds up the data loading process.
    early_stopping : bool
        Whether to use the early stopping rule defined in Optimization.train().
    from_scratch : bool (Default: False)
        - If True, trains the model from scratch.
        - If False, searches for trained model parameters with the same layer architecture,
        the same input, the same target variable, and the same hyperparameters. If they
        are available, continues training with these trained weigths and biases.

    Returns
    -------
    None
        Only saves model weights, performance, and runtime in the 
        respective folders.
    """
    
    # Start training process.
    print('Creating dataset for training and validation...')
    runtime_path = './runtime/train_models_runtime.pkl'
    
    # Create a DataSet object to extract data from if needed.
    if datatype == 'images' or datatype == 'images3d':
        if 'images' not in globals():
            images = data.create_dataset(
                type='images',
                from_pkl=True)
            dataset = images
        else:
            dataset = images
            
    elif datatype == 'irradiance':
        if 'irradiance' not in globals():
            irradiance = data.create_dataset(
                type='irradiance',
                from_pkl=True)
            dataset = irradiance
        else:
            dataset = irradiance

    elif datatype == 'weather':
        if 'weather' not in globals():
            weather = data.create_dataset(
                type='weather',
                from_pkl=True)
            dataset = weather
        else:
            dataset = weather

    elif datatype == 'combined':
        if 'combined' not in globals():
            combined = data.create_dataset(
                type='combined', 
                from_pkl=True)
            dataset = combined
        else:
            dataset = combined
    
    # Create a sample of the data if needed.
    if sample == True:
        try:
            dataset.sample[sample_size] = dataset.sample[sample_size] #check if it exists
        except KeyError:
            dataset.create_sample(size=sample_size, seed=42)

    # Iterate through each layer architecture.
    for layers in layers_list:

        # Read in the pickle file that contains the list 
        # of already indexed architectures for this datatype.
        # Only applies, if this file exists.
        try:
            layers_df = pd.read_pickle(f'./parameters/{datatype}/{datatype}_layers.pkl')
        
        # Otherwise, if it does not exist,
        # create the dataframe containing the first index and save it.
        except FileNotFoundError:

            # Create the directory if needed.
            layers_dir = os.path.dirname(f'./parameters/{datatype}/{datatype}_layers.pkl')
            if not os.path.exists(layers_dir):
                os.makedirs(layers_dir)

            # Save the df in that directory.
            layers_df = pd.DataFrame.from_dict(
                {0:[layers]},
                orient='index',
                columns=[f'{datatype}_layers'])
            layers_df.to_pickle(f'./parameters/{datatype}/{datatype}_layers.pkl')

        # Check if the current layer architecture has already been indexed.
        # For this, iterate through this df and compare each entry to the current architecture.
        layers_check = [str(layers) == str(i) for i in layers_df[f'{datatype}_layers']]

        # If it has been indexed before, get that index.
        if np.sum(layers_check) > 0:
            layers_idx = layers_check.index(True)

        # If it has not been indexed before, assign it the highest index + 1.
        # Then add it to the df and save this updated version.
        elif np.sum(layers_check) == 0:
            layers_idx = layers_df.index[-1] + 1
            
            layers_df = layers_df.append(
                {f'{datatype}_layers':layers},
                ignore_index=True)
            layers_df.to_pickle(f'./parameters/{datatype}/{datatype}_layers.pkl')

        print(f'The index of this architecture is {layers_idx}. You can find the architecture in ./parameters/{datatype}/{datatype}_layers.pkl.')

        # Iterate through all combinations of delta, length, and forecast horizon.
        for delta, length, horizon in itertools.product(
            delta_list,
            length_list,
            horizon_list):

            # Iterate through each combination of hyperparameters.
            for hyperparam in product_dict(hyperparameter_dict):
                
                # Create a name that includes the hyperparameters and its values.
                # This name is CRUCIAL as it is used throughout our analysis.
                # It serves as a unique identifier of a trained model within 
                # a particular datatype, input (delta, length), and forecast horizon.
                hyperparam_name = '__'.join([f'{k}({v})' for k, v in hyperparam.items()])
                
                # Check whether this architecture-delta-length-horizon-hyperparam
                # combination has already been trained.
                # For this, create the path that shall be checked.
                if datatype == 'skyimages':
                    delta = 0
                    length = 1
                
                filepath = f'/{datatype}/{datatype}_layers_{layers_idx}/delta_{delta}/' \
                            f'length_{length}/y_{horizon}/{hyperparam_name}'
                
                params_path = f'./parameters{filepath}__early_stopping(False).pt'
                params_dir = os.path.dirname(params_path)
                
                # Check if a file with the same hyperparam combination exists.
                try:
                    trained_param_files = os.listdir(params_dir)
                    existence_check = [hyperparam_name in i for i in trained_param_files]

                except FileNotFoundError:
                    existence_check = [0]

                # If such a file does exist, check whether the training of these
                # model parameters has been terminated due to early stopping.
                if np.sum(existence_check) > 0:

                    # Get the filename of the .pt file that matches our current hyperparams.
                    # Then extract the value of the early_stopping parameter.
                    existence_file = trained_param_files[existence_check.index(True)]
                    existence_name = existence_file.replace('.pt', '')
                    existence_hyperparams = getperformance.extract_hyperparameters(existence_name)

                    # If the early stopping parameter is True, we do not need to continue training
                    # this model-input-horizon-hyperparameter combination. Thus, we can also assign
                    # True to the skip_training variable.
                    skip_training = existence_hyperparams['early_stopping']
                    skip_training = str_to_bool(skip_training) #turns string to actual bool

                    # If the early stopping parameter is True but we also don't want to use
                    # early-stopping in this training round AND we dont't want to train from scratch,
                    # this means we most likely already trained the model using no early-stoppingrule.
                    # Thus, we can also skip training this model.       
                    if (skip_training == False) \
                        and (early_stopping == False) \
                        and (from_scratch == False):

                        skip_training = True
                
                # If such a file does not exist, we won't skip training.
                else:
                    existence_file = None
                    skip_training = False

                # Skip training if the model has already been trained until the
                # early stopping rule.
                if skip_training == True:

                    print('This architecture-delta-length-horizon-hyperparam combination has already ' 
                        'been trained until the early stopping rule.')
                    print('Moving onto next training...')
                    print('-------------------------------')

                    # Clean memory before skipping this training.
                    # This is only needed if we created new data in this iteration.
                    if 'X_train' in locals():
                        del X_train, X_val, X_test
                        del y_train, y_val, y_test
                        del t_train, t_val, t_test
                        gc.collect()
                        if GPU == True:
                            with torch.cuda.device('cuda:0'):
                                torch.cuda.empty_cache()
                    continue

                # We only need to extract and preprocess the X, y, data if we change
                # either the delta, length, horizon, or hyperparam values, i.e. when layers_only=False.
                # If layers_only=True, we can re-use the trainloaders from the previously
                # trained layer architecture. Thus, we can check if one of the dataloaders exist.
                # If the train_loader does not exist, but layers_only=True, it means that
                # we are currently in the first iteration of training different layer architectures.
                # Hence, the dataloaders must be created from scratch.
                if 'train_loader' not in locals():
                    print('Extracting and preprocessing the specified X, y data...')
                    
                    # Extract the correct data.
                    X, y, t = dataset.get_data(
                        datatype=datatype,
                        forecast_horizon=horizon,
                        sample=sample,
                        sample_size=sample_size,
                        delta=delta,
                        length=length)
                    
                    # Apply preprocessing and create train/val/test splits.
                    X_train, X_val, X_test, \
                        y_train, y_val, y_test, \
                        t_train, t_val, t_test = preprocessing.split_and_preprocess(
                            X, 
                            y,
                            t,
                            preprocessing=datatype)
                    
                    # Get input dimensions of X.
                    input_dim = X_train.shape[1:]

                    # Clean memory.
                    del X, y, t
                    gc.collect()
                
                # Check whether input data should be shuffled (CNN) or not (LSTM).
                if modeltype == 'LSTM':
                    images_bool = False
                else:
                    images_bool = True
                
                # Just as above, if we are only using this function to train
                # different layer architectures, i.e. layers_only=True,
                # we can re-use the dataloaders between iterations,
                # except in the very first iteration.
                if 'train_loader' not in locals():

                    train_loader, val_loader = preprocessing.create_dataloader(
                        X_train=X_train, 
                        X_val=X_val,
                        y_train=y_train, 
                        y_val=y_val,
                        batch_size=hyperparam['batch_size'],
                        images_bool=images_bool,
                        GPU=GPU) 
                    
                    # Clean memory.
                    del X_train, X_val, X_test
                    del y_train, y_val, y_test
                    del t_train, t_val, t_test
                    gc.collect()
                    if GPU == True:
                        with torch.cuda.device('cuda:0'):
                            torch.cuda.empty_cache()
                
                # Initialize model.
                if modeltype == 'CNN':
                    try:
                        model = cnnModel(
                            input_dim=input_dim,
                            **layers)
                    except RuntimeError:
                        print('This layer architecture is not possible as it shrinks the image too much.')
                        print('Moving onto next training...')
                        print('-------------------------------')
                        continue

                elif modeltype == 'LSTM':
                    model = lstmModel(
                        input_dim=input_dim,
                        **layers)
                    model = model.float()

                # Initialize optimization.
                # We will use Adam throughout our analysis.
                optimizer = Adam(
                    model.parameters(),
                    lr=hyperparam['lr'],
                    weight_decay=hyperparam['weight_decay'])

                opt = Optimization(
                    model=model, 
                    loss_fn=loss_fn, 
                    optimizer=optimizer,
                    GPU=GPU)

                # If we are not training from scratch, update the states of the
                # model and the optimizer using the states of the previous training.
                if existence_file is not None and from_scratch == False:
                    model, optimizer, previous_epoch = load_checkpoint(
                        model=model, optimizer=optimizer,
                        filename=params_path)
                    print('The model has already been trained but the early stopping rule '
                            'has not been reached, yet.')

                    # If we updated the model and optimizer states, we need to
                    # move the model and optimizer to the GPU if needed.
                    if GPU == True:

                        # Get the device from the train_loader.
                        for x, y in train_loader:
                            device = x.device
                            break
                        
                        # Move each state within the optimizer to the device.
                        for state in optimizer.state.values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.to(device)

                        # Also extract the losses from the previous training.
                        previous_losses = pd.read_pickle(
                            f'./performance/{filepath}.pkl')

                    # Indicate that we indeed do not want to train from scratch.
                    train_from_scratch = from_scratch

                # If there was no such file of trained parameters, indicate
                # that we are effectively training from scratch.
                else:
                    train_from_scratch = True
                    previous_losses = None
                    previous_epoch = 0
                
                # Train the model.
                opt.train(
                    train_loader=train_loader, 
                    val_loader=val_loader, 
                    filepath=filepath,
                    n_epochs=n_epochs,
                    GPU=GPU,
                    early_stopping=early_stopping,
                    from_scratch=train_from_scratch,
                    previous_losses=previous_losses)

                # Clean the memory if this function is NOT used to 
                # only train different model architectures.
                # Otherwise, we can re-use the same dataloaders.
                if layers_only == False:
                    del train_loader, val_loader, model, optimizer
                    gc.collect()
                    if GPU == True:
                        with torch.cuda.device('cuda:0'):
                            torch.cuda.empty_cache()
                
                # In either case, we must create a new model and optimizer.
                elif layers_only == True:
                    del model, optimizer
                    gc.collect()
                    if GPU == True:
                        with torch.cuda.device('cuda:0'):
                            torch.cuda.empty_cache()

                # create a df for the average runtime per epoch for forward pass, backprop, and eval.
                runtime_dict = {'Datatype':[datatype, datatype, datatype],
                                'Layers Index':[layers_idx, layers_idx, layers_idx],
                                'Delta':[delta, delta, delta],
                                'Length':[length, length, length],
                                'Horizon':[horizon, horizon, horizon],
                                'Hyperparameters':[hyperparam_name, hyperparam_name, hyperparam_name],
                                'Operation':['forward pass', 
                                                'backpropagation', 
                                                'evaluation'],
                                'Average Runtime Per Epoch (s)':[np.sum(opt.runtime_forward_training) / (n_epochs-previous_epoch),
                                                                np.sum(opt.runtime_backprop_training) / (n_epochs-previous_epoch),
                                                                np.sum(opt.runtime_evaluation) / (n_epochs-previous_epoch)]}
                new_runtime_df = pd.DataFrame.from_dict(runtime_dict)
                
                # Read in a df of the training runtime if it already exists.
                # Then add new runtime and save the whole df.
                try:
                    old_runtime_df = pd.read_pickle(runtime_path)
                    new_runtime_df = pd.concat([old_runtime_df, new_runtime_df],
                                                ignore_index=True)
                    new_runtime_df.to_pickle(runtime_path)

                # If it does not exist, just save this new df.
                except FileNotFoundError:
                    old_runtime_df = None
                    new_runtime_df.to_pickle(runtime_path)

                # Clean the memory.
                del opt, old_runtime_df, new_runtime_df
                gc.collect()
                if GPU == True:
                    with torch.cuda.device('cuda:0'):
                        torch.cuda.empty_cache()
                
                print('Finished training this architecture with:')
                print(f'- delta (X data): {delta} minutes')
                print(f'- length (X data): {length}')
                print(f'- horizon (y data): {horizon} minutes')
                print(f'- hyperparams: {hyperparam_name}')
                print('-------------------------------')


# Define a function used to train multiple NN ensemble models in one go.
def train_multiple_ensemble_models(GPU, layers_list, horizon_list, 
                                hyperparameter_dict, all_inputs, 
                                include_interactions=False, include_time=False, 
                                include_hour_poly=False, include_ghi=False, 
                                include_weather=False, from_scratch=False):
    """
    Function to apply the Optimization() class using an ensemble model 
    to different layer architectures,forecast horizons, and hyperparameters 
    in a row. This function is very similar to train_multiple_models(),
    but it is adapted to the specific inputs of the NN ensemble model.
    
    Parameters
    ----------
    GPU : bool: 
        Whether to train on the GPU (True) or on the CPU (False).
    layers_list : list 
        Array object including different model layer architectures.
    horizon_list : list
        Array object including different forecasting horizons of the target.
    hyperparameter_dict : dict
        Dictionary containing lists for each hyperparameter in this format:
        {'batch_size':[], 'lr':[], 'weight_decay':[]}
    all_inputs: bool
        - If yes, trains one ensemble for each ensemble data combinations,
        e.g. predictions, predictions+interactions, etc.
        - If no, only trains the model on the specified inputs using the
        include_xyz parameters such as include_interactions.
    include_interactions : bool (Default: False)
		If True, adds interaction terms of models predictions to the data.
	include_time : bool (Default: False)
		If True, adds date, month, and hour of the day at prediction time
		to the data.
	include_hour_poly : bool (Default: False):
		If True, adds the square and cube of the hour at prediction time 
		to the data.
	include_ghi : bool (Default: False)
		If True, adds irradiance to the data.
	Include_weather : bool (Default: False)
		If True, adds weather data to the data.
    from_scratch : bool (Default: False)
        - If True, trains the model from scratch.
        - If False, searches for trained model parameters with the same layer architecture,
        the same input, the same target variable, and the same hyperparameters. If they
        are available, continues training with these trained weigths and biases.

    Returns
    -------
    None
        Only saves model weights, performance, and runtime in the 
        respective folders.
    """

    # Create a list of all possible ensemble input data combinations, if needed.
    if all_inputs:
        
        # Denote the different datatypes that can be generated from data.get_ensemble_data().
        # Then add all possible data combinations to a list.
        ensemble_datatypes = ['interactions', 'time', 'hour_poly', 'ghi', 'weather']
        combos = [comb for i in range(len(ensemble_datatypes)) 
                    for comb in itertools.combinations(ensemble_datatypes, i + 1)]
        combos = [()] + combos
        combos = ['predictions__' + '__'.join(comb) for comb in combos]

    # Otherwise, we are only training on a single ensemble input combination.
    else:

        # Create a list of all the ensemble input data we want to train on.
        ensemble_datatypes = []
        if include_interactions:
            ensemble_datatypes.append('interactions')
        if include_time:
            ensemble_datatypes.append('time')
        if include_hour_poly:
            ensemble_datatypes.append('hour_poly'),
        if include_ghi:
            ensemble_datatypes.append('ghi')
        if include_weather:
            ensemble_datatypes.append('weather')
        combos = ['predictions__' + '__'.join(ensemble_datatypes)]
    
    # Iterate through each layer architecture.
    for layers in layers_list:

        # If it exists, read in the pickle file that contains the list 
        # of already indexed architectures of the ensemble NNs.
        try:
            layers_df = pd.read_pickle('./parameters/ensemble/ensemble_layers.pkl')
        
        # Otherwise, if the pickle file containing the indices of layer architectures does not exist
        # create the dataframe containing the first index and save it.
        except FileNotFoundError:

            # Create the directory if needed.
            layers_dir = os.path.dirname('./parameters/ensemble/ensemble_layers.pkl')
            if not os.path.exists(layers_dir):
                os.makedirs(layers_dir)

            # Save the df in that directory.
            layers_df = pd.DataFrame.from_dict({0:[layers]},
                                                orient='index',
                                                columns=['ensemble_layers'])
            layers_df.to_pickle(f'./parameters/ensemble/ensemble_layers.pkl')

        # Check if the current layer architecture has already been indexed.
        # For this, iterate through this df and compare each entry to the current architecture.
        layers_check = [str(layers) == str(i) for i in layers_df['ensemble_layers']]

        # If it has been indexed before, get that index.
        if np.sum(layers_check) > 0:
            layers_idx = layers_check.index(True)

        # If it has not been indexed before, assign it the highest index + 1,
        # then add it to the df and save this updated version.
        elif np.sum(layers_check) == 0:
            layers_idx = layers_df.index[-1] + 1
            
            layers_df = layers_df.append({'ensemble_layers':layers},
                                                ignore_index=True)
            layers_df.to_pickle(f'./parameters/ensemble/ensemble_layers.pkl')

        print(f'The index of this architecture is {layers_idx}. You can find the architecture in ./parameters/ensemble/ensemble_layers.pkl.')

        # Iterate through all specified ensemble data input combinations.
        for ensemble_name in combos:

            # Extract the parameters used for data.get_ensemble_data()
            ensemble_data_parameters = extract_ensemble_inputs(ensemble_name)
        
            # Iterate through each forecast horizon.
            for horizon in horizon_list:

                # Iterate through each combination of hyperparameters
                for hyperparam in product_dict(hyperparameter_dict):

                    # Create a name that includes the hyperparameters and its values.
                    hyperparam_name = '__'.join([f'{k}({v})' for k, v in hyperparam.items()])
                    
                    # Check whether this architecture-ensemble_input-horizon-hyperparam
                    # combination has already been trained.
                    # For this, create the path that shall be checked.
                    filepath = f'/ensemble/ensemble_layers_{layers_idx}/{ensemble_name}/' \
                                f'y_{horizon}/{hyperparam_name}'
                    
                    params_path = f'./parameters{filepath}__early_stopping(False).pt'
                    params_dir = os.path.dirname(params_path)
                    
                    # Check if a file with the same hyperparam combination exists.
                    try:
                        trained_param_files = os.listdir(params_dir)
                        existence_check = [hyperparam_name in i for i in trained_param_files]

                    except FileNotFoundError:
                        existence_check = [0]

                    # If such a file does exist, check whether the training of these
                    # model parameters has been terminated due to early stopping.
                    if np.sum(existence_check) > 0:

                        # Get the filename of the .pt file that matches our current hyperparams.
                        # Then extract the value of the early_stopping parameter.
                        existence_file = trained_param_files[existence_check.index(True)]
                        existence_name = existence_file.replace('.pt', '')
                        existence_hyperparams = getperformance.extract_hyperparameters(existence_name)

                        # If the early stopping parameter is True, we do not need to continue training
                        # this model-input-horizon-hyperparameter combination. Thus, we can also assign
                        # True to the skip_training variable.
                        skip_training = existence_hyperparams['early_stopping']
                        skip_training = str_to_bool(skip_training) #turns string to actual bool
                    
                    # If such a file does not exist, we won't skip training.
                    else:
                        existence_file = None
                        skip_training = False

                    # Skip training if the model has already been trained until the
                    # early stopping rule.
                    if skip_training == True:

                        print('This architecture-input-horizon-hyperparam combination has already ' 
                            'been trained until the early stopping rule.')
                        print('Moving onto next training...')
                        print('-------------------------------')

                        # Clean memory before skipping this training.
                        # This is only needed if we created new data in this iteration.
                        if 'X_train' in locals():
                            del X_train, X_val, X_test
                            del y_train, y_val, y_test
                            del t_train, t_val, t_test
                            gc.collect()
                            if GPU == True:
                                with torch.cuda.device('cuda:0'):
                                    torch.cuda.empty_cache()
                        continue

                    # Extract the correct ensemble data.
                    X_train, X_val, X_test, \
                        y_train, y_val, y_test, \
                        t_train, t_val, t_test, _ = data.get_ensemble_data(
                            horizon=horizon,
                            **ensemble_data_parameters)
                    
                    # Get input dimensions of X.
                    input_dim = X_train.shape[1]

                    # Create the data loaders.
                    train_loader, val_loader = preprocessing.create_dataloader(
                        X_train=torch.from_numpy(X_train).float(),
                        X_val=torch.from_numpy(X_val).float(),
                        y_train=torch.from_numpy(y_train).float(),
                        y_val=torch.from_numpy(y_val).float(),
                        batch_size=hyperparam['batch_size'],
                        images_bool=True, #we want to shuffle the data
                        GPU=GPU) 
                    
                    # Clean memory.
                    del X_train, X_val, X_test
                    del y_train, y_val, y_test
                    del t_train, t_val, t_test
                    gc.collect()
                    if GPU == True:
                        with torch.cuda.device('cuda:0'):
                            torch.cuda.empty_cache()

                    # Initialize model.
                    model = NN(input_dim=input_dim, **layers)
                    model = model.float()

                    # Initialize optimization.
                    optimizer = Adam(
                        model.parameters(),
                        lr=hyperparam['lr'],
                        weight_decay=hyperparam['weight_decay'])

                    opt = Optimization(
                        model=model, 
                        loss_fn=RMSELoss(), 
                        optimizer=optimizer,
                        GPU=GPU)

                    # If we are not training from scratch, update the states of the
                    # model and the optimizer using the states of the previous training.
                    if existence_file is not None and from_scratch == False:
                        model, optimizer, previous_epoch = load_checkpoint(
                            model=model, optimizer=optimizer,
                            filename=params_path)
                        print('The model has already been trained but the early stopping rule '
                                'has not been reached, yet.')

                        # If we updated the model and optimizer states, we need to
                        # move the model and optimizer to the GPU if needed.
                        if GPU == True:

                            # Get the device from the train_loader.
                            for x, y in train_loader:
                                device = x.device
                                break
                            
                            # Move each state within the optimizer to the device.
                            for state in optimizer.state.values():
                                for k, v in state.items():
                                    if isinstance(v, torch.Tensor):
                                        state[k] = v.to(device)

                            # Also extract the losses from the previous training.
                            previous_losses = pd.read_pickle(
                                f'./performance/{filepath}.pkl')

                        # Indicate that we indeed do not want to train from scratch.
                        train_from_scratch = from_scratch

                    # If there was no such file of trained parameters, indicate
                    # that we are effectively training from scratch.
                    else:
                        train_from_scratch = True
                        previous_losses = None
                        previous_epoch = 0

                    # Train the model
                    opt.train(train_loader=train_loader, 
                        val_loader=val_loader, 
                        filepath=filepath,
                        n_epochs=500, #large number to ensure convergence
                        GPU=GPU,
                        early_stopping=True, #utlilize the early stopping rule
                        from_scratch=train_from_scratch,
                        previous_losses=previous_losses)

                    # Clean the memory.
                    del train_loader, val_loader, model, optimizer, opt
                    gc.collect()
                    if GPU == True:
                        with torch.cuda.device('cuda:0'):
                            torch.cuda.empty_cache()

                    print('Finished training this ensemble architecture with:')
                    print(f'- input (X data): {ensemble_name}')
                    print(f'- horizon (y data): {horizon} minutes')
                    print(f'- hyperparams: {hyperparam_name}')
                    print('-------------------------------')


# Define functin to load a trained model by selecting
# its architecture, input data, forecast horizon, and hyperparameters
# that were used for training.
def load_trained_model(datatype, layers_idx, delta, length, horizon,
                       batch_size, lr, weight_decay):
    """
    Function that first selects the correct model (CNN or LSTM) for the datatype.
    It then searches the respective layer architecture for the given layers_idx.
    It then loads in the model parameters based on the X data (delta, length),
    the y data (horizon) and hyperparameters used to train the model. 
    Then, this specific model is trained.

    Parameters
    ----------
    datatype : {'images', 'images3d', 'irradiance', 'weather', 'combined'}
        Which data type the model was trained on.
    layers_idx : int 
        Unique index that specifies the layer architecture.
        The index can be found in the respective directory per datatype,
        e.g. in ./parameters/images3d/images3d_layers.pkl
    delta : int 
        The delta (in minutes) of the sequence used to train the model.
        delta=0 for datatype='images'.
    length : int 
        The length of the sequence used to train the model.
        length=1 for datatype='images'.
    horizon : int 
        The forecast horizon (in minutes) of the target used to train the model.
    batch_size : int 
        Batch size used to train the model.
    lr : float 
        Learning rate used to train the model.
    weight_decay : float 
        Weight decay used to train the model
    
    Returns
    -------
    model : torch.nn.Module
        Returns the trained PyTorch model incl. weights and biases 
        that was trained on the specified parameters.
    """

    # Get the layer architecture corresponding to the specified layers_idx.
    layers_df = pd.read_pickle(f'./parameters/{datatype}/{datatype}_layers.pkl')
    layers = layers_df.loc[layers_idx, f'{datatype}_layers']

    # Load the correct model based on datatype and the specified layer architecture.
    if datatype == 'images':
        input_dim = (1, 200, 200)
        model = cnnModel(input_dim, **layers)

    elif datatype == 'images3d':
        input_dim = (1, length, 200, 200)
        model = cnnModel(input_dim, **layers)

    elif datatype == 'irradiance':
        input_dim = (25, 1) #only the number of columns (1) is relevant here
        model = lstmModel(input_dim, **layers)

    elif datatype == 'weather':
        input_dim = (25, 7) #only the number of columns (7) is relevant here
        model = lstmModel(input_dim, **layers)

    elif datatype == 'combined':
        input_dim = (25, 8) #only the number of columns (8) is relevant here
        model = lstmModel(input_dim, **layers)
    
    # Define the filename of the model parameters.
    file_name = f'batch_size({batch_size})__lr({lr})__weight_decay({weight_decay})'
    
    # Define the path where the model parameters are stored.
    earlystopping_path = f'./parameters/{datatype}/{datatype}_layers_{layers_idx}/delta_{delta}/' \
                        f'length_{length}/y_{horizon}/{file_name}__early_stopping(True).pt'
    normalstopping_path = f'./parameters/{datatype}/{datatype}_layers_{layers_idx}/delta_{delta}/' \
                        f'length_{length}/y_{horizon}/{file_name}__early_stopping(False).pt'
    
    # Load the trained parameters into the model.
    # First, try to load the model in case the early stopping rule has been reached.
    try:
        checkpoint = torch.load(earlystopping_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    
    # The models trained in the first step of the training process, 
    # i.e. the step where we just varied the layer architectures
    # only had their parameters saved and not the state of the optimizer + last epoch.
    # Hence, for these models we need to directly load in the model state dict.
    except KeyError:
        model.load_state_dict(torch.load(earlystopping_path, map_location='cpu'))

    # The other possibility is that the path does not exist, i.e. that the
    # early stopping rule has not been reached for this particular model.
    except FileNotFoundError:

        # Here, we again have two possibilites:
        # One, the model parameters were saved together with the optimizer and epoch:
        try:
            checkpoint = torch.load(normalstopping_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])

        # Or the model parameters were saved on their own.
        except KeyError:
            model.load_state_dict(torch.load(normalstopping_path, map_location='cpu'))
            
    # Set it to evaluation mode and return it.
    return model.eval()


# Create a function that generates a list of dictionaries
# based on possible CNN parameters that should be tried out
# where the dictionaries are all the possible CNN parameter
# permutations that are used to initialize the model.
def create_CNN_permutations(all_cnn_num, all_conv_filters, all_conv_kernels, 
                            all_conv_strides, conv_padding,
                            maxpool_kernel, maxpool_stride, 
                            maxpool_padding, all_dense_num, 
                            all_dense_dropout, dense_activation, 
                            dense_hidden_size):
    """
    Function that takes in a list for each CNN parameter 
    and returns a list of dictionaries for many sensible model parameter 
    permutations. This list of dictionaries can then be fed into
    the layers_list argument in the train_all_models() function.

    Parameters
    -----------
    all_cnn_num : list
        All possible integer values of number
        of convolutional layers that should be tested.
    all_conv_filters : list
        All possible integer values of number
        of convolutional filters that should be tested.
    all_conv_kernels : list
        All possible integer values of filter sizes
        that should be tested.
    all_conv_strides : list
        All possible integer values of stride sizes
        of the conv. filters that should be tested.
    conv_padding : int
        Padding around matrix before application of
        conv. filter that should be used for all
        of the layer architectures.
    maxpool_kernel : int
        Kernel size of the maxpool operations
        that should be used for all architectures.
    maxpool_stride : int
        Stride of the maxpool operations that should
        be used for all architectures.
    maxpool_padding : int
        Size of the padding around the matrix before
        the maxpool operation that should be used for
        all architectures.
    all_dense_num : list
        All possible integer values for the number of
        dense layers that are fed the output of
        the convolutional layers.
    all_dense_dropout : list
        All possible float values for the dropout layers
        of the dense layers.
    dense_activation : torch.nn obj
        Which activation function to use within the
        dense layer(s).
    dense_hidden_size : int
        Number of hidden neurons within the dense layer(s)
        that should be used for all architectures.

    Returns
    -------
    lstm_layers_list : list
        List of dictionaries containing the parameters
        that are used to initialize the cnnModel().
    """

    # Check whether these respective inputs are integers
    # as otherwise the number of permutations will be too high:
    if not isinstance(conv_padding, int):
        raise TypeError('con_padding must be an integer. Otherwise, number of permutations will be too large.')
    if not isinstance(maxpool_kernel, int):
        raise TypeError('maxpool_kernel must be an integer. Otherwise, number of permutations will be too large.')
    if not isinstance(maxpool_stride, int):
        raise TypeError('maxpool_stride must be an integer. Otherwise, number of permutations will be too large.')
    if not isinstance(maxpool_padding, int):
        raise TypeError('maxpool_padding must be an integer. Otherwise, number of permutations will be too large.')
    if not isinstance(maxpool_padding, int):
        raise TypeError('maxpool_padding must be an integer. Otherwise, number of permutations will be too large.')
    if not isinstance(dense_hidden_size, int):
        raise TypeError('dense_hidden_size must be an integer. Otherwise, number of permutations will be too large.')

    # Initialize the list output.
    cnn_layers_list = []
    
    # It would not be wise to choose literally all possible permuations
    # as this would ensure the number of model parameter permutations
    # to quickly soar above 10,000 or even 100,000.
    # Hence we will only try number of convolutional filters that are
    # either strictly increasing from layer to layer or that are
    # staying the same from layer to layer. Also, we will only use
    # permutations where the filter kernels and strides are 
    # decreasing from layer to layer.

    # Iterate through all possible number of cnn layers.
    for cnn_num in all_cnn_num:

        # Get all permutations of convolutional filters of length cnn_num
        # that are either strictly increasing or all equal
        prod = itertools.product(all_conv_filters, repeat=cnn_num)
        perm_conv_filters = [p for p in prod if strictly_decreasing(p)==True \
                                            or len(set(p))==1]

        # Get all permutations of convolutions filter kernels and strides
        # that are strictly decreasing from layer to layer.
        prod = itertools.product(all_conv_kernels, repeat=cnn_num)
        perm_conv_kernels = [p for p in prod if strictly_decreasing(p)==True]
        prod = itertools.product(all_conv_strides, repeat=cnn_num)
        perm_conv_strides = [p for p in prod if strictly_decreasing(p)==True]

        # However, if the number of layers (cnn_num) is larger than the given
        # number of possible kernel or stride values, then it is not possible to build 
        # such a strictly decreasing list. In that case, build a strictly decreasing
        # list containing all possible kernel/stride values and fill up the list
        # with the lowest kernel/stride values such that its length is equal
        # to the number of layers. This ensures that we have one kernel/stride 
        # for each layer.
        if len(perm_conv_kernels) == 0:
            prod = itertools.product(all_conv_kernels, repeat=len(all_conv_kernels))
            perm = [i for i in prod if strictly_decreasing(i)==True]
            
            # Add the lowest kernels to the strictly decreasing list
            perm_conv_kernels = [*perm[0]] + (cnn_num-len(all_conv_kernels))*[min(all_conv_kernels)]
            perm_conv_kernels = [tuple(perm_conv_kernels)]

        # Repeat for the filter strides.
        if len(perm_conv_strides) == 0:
            prod = itertools.product(all_conv_strides, repeat=len(all_conv_strides))
            perm = [i for i in prod if strictly_decreasing(i)==True]
            
            # Add the lowest kernels to the strictly decreasing list.
            perm_conv_strides = [*perm[0]] + (cnn_num-len(all_conv_strides))*[min(all_conv_strides)]
            perm_conv_strides = [tuple(perm_conv_strides)]

        # We now also need to create all possible dense layer permutations.
        perm_dense = []

        # For this, we iterate through all given number of possible dense layers.
        for dense_num in all_dense_num:

            # If we only have one dense layer, we do not need the dropout layer or 
            # the number of hidden neurons. Hence, we only have one possible architecture.
            if dense_num == 1:
                perm_dense.append({'dense_num':dense_num,
                                    'dense_dropout':0,
                                    'dense_activation':nn.ReLU(),
                                    'dense_hidden_size':0})
                continue

            # For all other number of dense layers, the dropout layer
            # and the number of hidden units are relevant.
            # Hence, we will add all possible values to the list.
            if dense_num > 1:

                for drop in all_dense_dropout:

                    perm_dense.append({'dense_num':dense_num,
                                    'dense_dropout':drop,
                                    'dense_activation':nn.ReLU(),
                                    'dense_hidden_size':dense_hidden_size})
        
        # We can now create all possible convolutional layers based on these
        # selected permuations and using the given parameters, such as
        # maxpool_padding.
        for conv_filters, conv_kernels, conv_strides, dense in itertools.product(
                                                                perm_conv_filters,
                                                                perm_conv_kernels,
                                                                perm_conv_strides,
                                                                perm_dense):
            cnn_layers_list.append({'cnn_num':cnn_num,
                                    'conv_filters':list(conv_filters),
                                    'conv_kernels':list(conv_kernels),
                                    'conv_strides':list(conv_strides),
                                    'conv_paddings':cnn_num * [conv_padding],
                                    'maxpool_kernels':cnn_num * [maxpool_kernel],
                                    'maxpool_strides':cnn_num * [maxpool_stride],
                                    'maxpool_paddings':cnn_num * [maxpool_padding],
                                    **dense})

    # We now have a list containing dictionary of parameters that can be 
    # used to initialize a CNN model. 
    print(f'You have selected {len(cnn_layers_list)} different model architectures for our CNNs.')
    if len(cnn_layers_list) > 100:
        print('Training these will take a while.')
    
    return cnn_layers_list


# Create a function that generates a list of dictionaries
# based on possible LSTM parameters that should be tried out
# where the dictionaries are all the possible LSTM parameter
# permutations that are used to initialize the model.
def create_LSTM_permutations(all_lstm_hidden_size, all_lstm_num_layers, 
                            all_lstm_dropout, all_dense_num, 
                            all_dense_dropout, dense_activation, 
                            dense_hidden_size):
    """
    Function that takes in a list for each LSTM parameter 
    and returns a list of dictionaries for many sensible model parameter 
    permutations. This list can then be fed to the 
    layers_list argument in the train_all_models() function.

    Parameters
    ----------
    all_lstm_hidden_size : list
        All possible integer values for the number of 
        hidden neurons within the LSTM layer.
    all_lstm_num_layers : list
        All possible integer values for the number of 
        layers of the LSTM.
    all_lstm_dropout : list
        All possible float values for the dropout
        layer of the LSTM.
    all_dense_num : list
        All possible integer values for the number of
        dense layers that are fed the output of
        the LSTM layers.
    all_dense_dropout : list
        All possible float values for the dropout layers
        of the dense layers.
    dense_activation : torch.nn obj
        Which activation function to use within the
        dense layer(s).
    dense_hidden_size : int
        Number of hidden neurons within the dense layer(s).

    Returns
    -------
    lstm_layers_list : list
        List of dictionaries containing the parameters
        that are used to initialize the lstmModel().
    """

    # Lets's firrst create all possible dense layer permutations.
    perm_dense = []

    # For this, we iterate through all given number of possible dense layers.
    for dense_num in all_dense_num:

        # If we only have one dense layer, we do not need the dropout layer or 
        # the number of hidden neurons. Hence, we only have one possible architecture.
        if dense_num == 1:
            perm_dense.append({'dense_num':dense_num,
                                'dense_dropout':0,
                                'dense_activation':nn.ReLU(),
                                'dense_hidden_size':0})
            continue

        # For all other number of dense layers, the dropout layer
        # and the number of hidden units are relevant.
        # Hence, we will add all possible values to the list.
        if dense_num > 1:

            for drop in all_dense_dropout:

                perm_dense.append({'dense_num':dense_num,
                                'dense_dropout':drop,
                                'dense_activation':nn.ReLU(),
                                'dense_hidden_size':dense_hidden_size})
    
    # Initialize the final output.
    lstm_layers_list = []

    # Iterate through all possible permutations of the parameters
    # that initialize the lstmModel().
    for (lstm_hidden_size, lstm_num_layers, 
        lstm_dropout, dense) in itertools.product(
            all_lstm_hidden_size,
            all_lstm_num_layers,
            all_lstm_dropout,
            perm_dense):

        # Add each permuation to a dictionary and append it to the output.
        lstm_layers_list.append(
            {'lstm_hidden_size':lstm_hidden_size,
            'lstm_num_layers':lstm_num_layers,
            'lstm_dropout':lstm_dropout,
            **dense})

    return lstm_layers_list
    

# Define a function that creates all possible combinations of NN() inputs.
def create_NN_combinations(all_hidden_num, all_hidden_size, all_dropout_prob):
    """
    Function that creates a dictionary with 'hidden_num', 'hidden_size', and
    'dropout_prob' as keys and the respective values for each combination of 
    the given inputs. Used to construct the NN() class object.

    Parameters
    ----------
    all_hidden_num : list
        List containing all possible integer values that should be used for the 
        number of hidden units of the NN layer architecture.
    all_hidden_size : list
        List containing all possible integer values that should be used for the 
        size of the hidden layers of the NN layer architecture.
    all_dropout_probs
        List containing all possible float values that should be used for the 
        dropout probabilities of the NN layer architecture.

    Returns
    -------
    nn_combinations : list
        List of dictionaries containing the parameters
        that are used to initialize the model NN().
    """

    # Initialize output.
    nn_combinations = []
    
    # Iterate through all combinations and add them to a dictionary.
    for hidden_num, hidden_size, dropout_prob in itertools.product(
        all_hidden_num, all_hidden_size, all_dropout_prob):

        # Create the dict and add it to the output.
        layer_dict = {
            'hidden_num' : hidden_num,
            'hidden_size' : hidden_size,
            'dropout_prob' : dropout_prob}
        nn_combinations.append(layer_dict)

    return nn_combinations


# Define a function to load in a pre-trained model to continue training.
def load_checkpoint(model, optimizer, filename):
    """
    Function that reads in a checkpoint of trained model parameters.

    Parameters
    ----------
    optimizer : torch.optim obj
        Torch optimizer object, such as torch.optim.Adam.
    filename : str
        Filename of the model checkpoint.

    Returns
    -------
    model : torch.nn.Module
        Pytorch model object.
    optimizer : torch.optim obj
        Torch optimizer object, such as torch.optim.Adam
        that contains information about previous training epochs.
    start_epoch : int
        At which epoch to start the training.
    """

    # Only update state of model and optimizer if the file exists.
    start_epoch = 0
    if os.path.isfile(filename):
        print('Loading in already trained model to continue training.')
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'Loaded checkpoint at epoch {start_epoch})')
    else:
        print(f'No checkpoint found at {filename}')

    return model, optimizer, start_epoch


# Define a function that checks whether all values in a list are strictly decreasing.
# Used in create_CNN_permutations().
def strictly_decreasing(lst):
    return all(x>y for x, y in zip(lst, lst[1:]))


# Define a function that checks whether all values in a list are strictly increasing.
# Used in create_CNN_permutations().
def strictly_increasing(lst):
    return all(x<y for x, y in zip(lst, lst[1:]))


# Define function to iterate through all key-value pairs in a dictionary.
# Source: https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def product_dict(d):
    return (dict(zip(d.keys(), values)) for values in itertools.product(*d.values()))


# Define a function that turns a boolean string into an actual boolean.
# Used to check whether trained model parameters have reached the early-stopping rule.
def str_to_bool(string):
    if string == 'True':
         return True
    elif string == 'False':
         return False


# Define RSMELoss() criterion as it does not exist in PyTorch.
class RMSELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.epsilon = epsilon #in case of MSE = 0
        
    def forward(self, yhat, y):
        
        # return the square root of the MSE (+ some small episolon)
        loss = torch.sqrt(self.mse(yhat,y) + self.epsilon)
        return loss


# Define a function that extracts the relevant parameters for data.get_ensemble_data()
# from a filepath containing the ensemble model inputs.
def extract_ensemble_inputs(ensemble_name):
    """
    Function that takes in a string such as predictions__interactions__time that is
    used to store ensemble models and extracts the relevant parameters needed to 
    create ensemble data using data.get_ensemble_data(), e.g. include_time=True.

    Parameters
    ----------
    ensemble_name : str
        String used to store ensemble models in the correct folder, 
        e.g. predictions__interactions__time.

    Returns
    -------
    ensemble_data_parameters : dict
        Dictionary containing the parameters for data.get_ensemble_data()
        and their respective boolean values, e.g. 'include_time' : True.
    """

    # Split the string and remove 'predictions' as these are included in every model.
    parameters_string = ensemble_name.replace('predictions__', '')
    parameters = parameters_string.split('__')

    # Create a dictionary containing the necessary parameters for data.get_ensemble_data()
    # and fill the boolean values based on their existence in the list of parameters.
    ensemble_data_parameters = {
        'include_interactions' : 'interactions' in parameters,
        'include_time' : 'time' in parameters, 
		'include_hour_poly' : 'hour_poly' in parameters, 
        'include_ghi' : 'ghi' in parameters, 
        'include_weather' : 'weather' in parameters}

    return ensemble_data_parameters


