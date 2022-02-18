"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
getpeformance.py defines the functions
that are used to compare the forecast performances 
among different models that we want to compare.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# Import outside modules
import os
import gc
import itertools
import weakref
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import torch

# Import our own modules
import models
import data
import preprocessing


# Define a function that aggregates the validation set performance
# of many different model layer architectures.
def get_layers_performance(datatype, delta, length, horizon, batch_size, lr, weight_decay,
                            min_or_mean, output_missing=True):
    """
    Function that checks the validation set performance for all
    layer architectures of a model trained on a specific datatype.
    It outputs the best performing layer architecture
    for a given forecast horizon (y), delta/length of the input (X), 
    and hyperparameters.
    
    Parameters
    ----------
    datatype : {'images', 'images3d', 'irradiance', 'weather', 'combined'}
        Which datatype was used to train the models whose
        performances are to be checked.
    delta : int
        Which delta (in minutes) was between the datapoints
        of the X data sequence used to train the models.
    length : int
        How long the sequence of the X data was that was 
        used to train the models.
    horizon : int
        The forecast horizon of the y target the models were 
        trained on.
    batch_size : int
        What batch sizes were used to train the models.
    lr : float
        What learning rates were used to train the models.
    weight_decay : float
        What weight decay was used to train the models.
    min_or_mean : {'min', 'mean'}
        - If min, aggregates the RMSEs using the best RMSEs per model.
        - If mean, aggregates the RMSEs using the mean of the lowest 5 RMSES
        per model.
    output_missing : bool (Default: True)
        Whether to print how many trained models where not found
        given the specified parameters.
    """

    # Check which layer architectures exist for the specified datatype
    # and initialize the performance for each architecture.
    layers = pd.read_pickle(f'./parameters/{datatype}/{datatype}_layers.pkl')
    performance = []

    # Create the path and the filename that should be searched.
    perf_path = f'./performance/{datatype}'
    name = f'batch_size({batch_size})__lr({lr})__weight_decay({weight_decay})'
    
    # Iterate through each layers index and get the training performance
    # for the specified delta, length, horizon, and hyperparameters.
    missing = []
    for i in layers.index:

        # We will use the mean of the 5 minimum validation losses
        # and define those as the 'Best Validation RMSE' to account
        # for variation between epochs. 
        # Only needed if min_or_mean == 'mean'
        num_min_rmse = 5
        
        # Specify the correct path and get the dataframe. Extract the RMSE if possible.
        # Otherwise, the performance will be NaN.
        path = f'{perf_path}/{datatype}_layers_{i}/delta_{delta}/length_{length}/y_{horizon}/{name}.pkl'
        try:
            df = pd.read_pickle(path)
            
            # Either get the best or the mean of the 5 best RMSEs.
            if min_or_mean == 'min':
                performance.append(df['val'].min())
            elif min_or_mean == 'mean':
                performance.append(df['val'].nsmallest(num_min_rmse).mean())
        except (FileNotFoundError, EOFError) as e:
            missing.append(i)
            performance.append(np.nan)
        
    # Add the delta, length, horizon, and hyperparameters to the
    # dataframe containing the layer architectures.
    for col, val in zip(['delta', 'length', 'horizon', 'batch_size', 'lr', 'weight_decay'],
                        [delta, length, horizon, batch_size, lr, weight_decay]):
        layers[col] = (len(layers)) * [val]

    # Also add the best performance for each architecture and rename the layers column.
    layers['Best Validation RMSE'] = performance
    layers.rename(columns={f'{datatype}_layers':'layers'}, inplace=True)

    if output_missing == True:
        print(f'Was not able to collect the performance of {len(missing)} '
                f'out of {len(layers.index)} total layer architectures for datatype {datatype}.')

    return layers
    

# Define a function that outputs the best performing layer architecture
# for a given datatype, input, target, and hyperparmeters.
def get_best_layers(datatype, return_layer_params, delta, length, horizon,
                    batch_size, lr, weight_decay, output_missing=True):
    """
    Function that uses get_layers_performance() to get the performance
    of all the layer architectures of a given datatype, delta, length,
    etc. It then outputs the layer parameters (or just the
    layer idx) of the best performing architecture. This can then be 
    used to initalize the respective model (CNN or LSTM).
    
    Parameters
    ----------
    datatype : {'images', 'images3d', 'irradiance', 'weather', 'combined'}
        Which datatype was used to train the models whose
        layers are to be returned.
    return_layer_params : bool
        - If True, returns all parameters of the layers.
        - If False, returns just the index of the layers.
    delta : int
        Which delta (in minutes) was between the datapoints
        of the X data sequence used to train the models.
    length : int
        How long the sequence of the X data was that was 
        used to train the models.
    horizon : int
        The forecast horizon of the y target the models were 
        trained on.
    batch_size : int
        What batch sizes were used to train the models.
    lr : float
        What learning rates were used to train the models.
    weight_decay : float
        What weight decay was used to train the models.
    output_missing : bool (Default: True)
        Whether to print how many trained models were not found
        given the specified parameters.

    Returns 
    -------
    best_layers : dict
        - If return_layer_params=True, returns layer
        parameters of the best architecture.
    best_idx : int
        - If return_layer_params=False, returns index
        of the best performing architecture.
    """

    # Get the RSMEs of all the layer architectures for the 
    # specified parameters.
    layers_df = get_layers_performance(
        datatype=datatype,
        delta=delta,
        length=length,
        horizon=horizon,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        output_missing=output_missing,
        min_or_mean='mean') #we used no early stopping when training
                            #the different layer architectures
                            #so we should aggregate the RMSEs by using
                            #the mean of the lowest RMSEs per model.

    # Output the layer parameters or the layers index
    # that correspond to the best RMSE.
    best_idx = layers_df['Best Validation RMSE'].idxmin()

    if return_layer_params == True:
        best_layers = layers_df.loc[best_idx, 'layers']
        return best_layers
    elif return_layer_params == False:
        return best_idx


# Define a function that returns the model and input
# parameters of the best trained model for a given
# datatype and forecast horizon.
def get_best_model_by_horizon(datatype, horizon, return_delta_length, 
                            return_hyperparams, batch_size, lr, weight_decay):
    """
    Function that returns all the necessary parameters,
    i.e. layer parameters, delta, length, and hyperparams,
    to load in the best model for a given datatype and
    forecast horizon.

    Parameters
    ----------
    datatype : {'images', 'images3d', 'irradiance', 'weather', 'combined'}
        Which datatype the model was trained with.
    horizon : int
        What forecast horizon the model was trained on.
    return_delta_length : bool
        Whether to also a return the tuple (delta, length) containing the
        respective values for delta and length of the input data
        that was used to train the model that is returned.
    return_hyperparams : bool
        Whether to also return a dictionary containing the values of the
        hyperparameters of the the model that is returned.
    batch_size : None or int
        - If None, will compare the RMSEs of all possible batch sizes.
        - If int, will compare the RMSEs of this specified batch size.
    lr : None or float
        - If None, will compare the RMSEs of all possible learning rates.
        - If float, will compare the RMSEs of this specified learning rates.
    weight_decay : None or float
        - If None, will compare the RMSEs of all possible weight decays
        - If float, will compare the RMSEs of this specified weight decay.
    
    Returns
    -------
    model : torch.nn.Module
        Loaded PyTorch model that has the best validation RMSE
        across all layer architectures, deltas, lengths, and 
        hyperparameters for a given datatype and horizon.
    delta : int
        Also returns the delta of the input data used to train the model.
        Only if return_delta_length=True.
    length : int
        Also returns the length of the input data used to train the model.
        Only if return_delta_length=True.
    hyperparams : dict
        Also returns a dictionary of hyperparameters used to train the model.
        Only if return_hyperparams=True.
    """

    # We have trained the models on three forecast horizon buckets:
    # intra-hour, intra-day, and day-ahead. We used a horizon of
    # 30min, 2h, and 1d for each of these buckets respectively.
    # Thus, if this function gets a horizon that is not one of these
    # three, we need to change it to the correct value of the bucket.
    if horizon < 60:
        horizon = 30 #change all such horizons to the 30min used for training
                        #of the intra-hour models
    elif horizon < 60*24:
        horizon = 120 #change all such horizons to the 2h used for training
                        #of the intra-day models
    else:
        horizon = 1440 #change all such horizons to the 1d used for training
                        #of the day-ahead models
    
    # Define a dictionary of the deltas and lengths that should be
    # searched for each datatype.
    data_dict = {'images':{'delta':[0],
                            'length':[1]},
                'images3d':{'delta':[10, 30, 60],
                            'length':[3]},
                'irradiance':{'delta':[1, 3, 5, 10, 15],
                            'length':[10, 25, 50, 100]},
                'weather':{'delta':[1, 3, 5, 10, 15],
                            'length':[10, 25, 50, 100]},
                'combined':{'delta':[1, 3, 5, 10, 15],
                            'length':[10, 25, 50, 100]}}

    # Define a dictionary of the parameters that we used in the very
    # first training step, i.e. the params used to find the best
    # layer architecture per datatype.
    layer_dict = {'images':{'delta':0,
                            'length':1,
                            'horizon':60,
                            'batch_size':128,
                            'lr':1e-04,
                            'weight_decay':1e-05},
                'images3d':{'delta':10,
                            'length':3,
                            'horizon':60,
                            'batch_size':64, #was needed due to GPU constrains
                            'lr':1e-04,
                            'weight_decay':1e-05},
                'irradiance':{'delta':3, #we used a delta=3 for all LSTMs
                            'length':25, #also same for all LSTMs
                            'horizon':60,
                            'batch_size':128,
                            'lr':1e-04,
                            'weight_decay':1e-05},
                'weather':{'delta':3, 
                            'length':25,
                            'horizon':60,
                            'batch_size':128,
                            'lr':1e-04,
                            'weight_decay':1e-05},
                'combined':{'delta':3, 
                            'length':25,
                            'horizon':60,
                            'batch_size':128,
                            'lr':1e-04,
                            'weight_decay':1e-05}}

    # Get the layers index to find the best performing
    # layer architecture per datatype.
    layers_idx = get_best_layers(
        datatype=datatype, 
        return_layer_params=False, 
        **layer_dict[datatype],
        output_missing=False)
    
    # Prepare the RMSE benchmark to compare the different models.
    benchmark = {'rmse':10000000000, #initialize at large value
                'delta':None,
                'length':None,
                'hyperparams':None}
    
    # Iterate through all different deltas and lengths.
    for delta, length in itertools.product(
        data_dict[datatype]['delta'],
        data_dict[datatype]['length']):
        
        # Enter path of the respective model and get all 
        # hyperparmater combinations that were trained.
        path = f'./performance/{datatype}/{datatype}_layers_{layers_idx}/' \
                f'delta_{delta}/length_{length}/y_{horizon}/'
        try:
            loss_dfs = [f for f in os.listdir(path) if not f.startswith('.')
                        and 'final' not in f] #ensures that the final model trained on
                                                #100% of the data is excluded here
        except FileNotFoundError:
            continue

        # Read in the dataframe for every hyperparameter combination
        # or for just the specified ones.
        for loss_df in loss_dfs:

            # Extract the hyperparameters from the name.
            hyperparams = extract_hyperparameters(
                hyperparam_name=loss_df.replace('.pkl', ''),
                as_string=False)

            # Check if the hyperparameters of this particular file
            # are among the specified ones in the function parameters.
            # If None, that means we want to consider all values
            # for this particular hyperparameter.

            if batch_size is not None:
                if hyperparams['batch_size'] != batch_size:
                    continue
            
            if lr is not None:
                if hyperparams['lr'] != lr:
                    continue

            if weight_decay is not None:
                if hyperparams['weight_decay'] != weight_decay:
                    continue
                
            # If all hyperparameters of this particular file passed
            # this "filter", then we can read in the df
            # containing the RMSEs.
            df = pd.read_pickle(path+loss_df)

            # Check if the best mini-batch validation RMSE of this 
            # particular model is better than the current best performance. 
            # If yes, add the information to the benchmark.
            best_rmse = df['val'].min()

            if best_rmse < benchmark['rmse']:
                benchmark['rmse'] = best_rmse
                benchmark['delta'] = delta
                benchmark['length'] = length
                benchmark['hyperparams'] = hyperparams
    
    # Load the correct model based on the parameters that yield
    # the best mini-batch validation RMSE.
    if benchmark['hyperparams'] is not None:
        model = models.load_trained_model(
            datatype=datatype, 
            layers_idx=layers_idx, 
            delta=benchmark['delta'], 
            length=benchmark['length'], 
            horizon=horizon,
            **benchmark['hyperparams'])
    else:
        model = None
    
    # Return the model and other infos about the model if specified.
    if return_delta_length == True and return_hyperparams == True:
        return model, benchmark['delta'], benchmark['length'], benchmark['hyperparams']

    elif return_delta_length == True and return_hyperparams == False:
        return model, benchmark['delta'], benchmark['length']

    elif return_delta_length == False and return_hyperparams == True:
        return model, benchmark['hyperparams']

    else:
        return model


# Define a function that aggregates the best layer architecture
# and the corresponding RMSE per datatype.
def aggregate_best_layers(datatypes, deltas, lengths, horizon, 
                        batch_sizes, lrs, weight_decays):
    """
    Function that creates a dataframe containing the layer architecture
    of the best performing architecture per datatype.

    Parameters
    ----------
    datatypes : str or 1d-array
                {'images', 'images3d', 'irradiance', 'weather', 'combined'}
        - If str, the dataframe will only contain the layer architecture
        for one datatype.
        - If 1d-array, the dataframe will contain the layer architecture
        for multiple datatypes.
    deltas : int or 1d-array
       - If int, indicates which delta was used for the input data
       of the given datatype model.
       - If 1d-array, indicates which delta was used for each of
       the input data of the given datatypes.
    lengths : int or 1d-array
       - If int, indicates which length was used for the input data
       of the given datatype model.
       - If 1d-array, indicates which length was used for each of
       the input data of the given datatypes.
    horizon : int
        The forecast horizon (in minutes) of the y target 
        the models were trained on.
    batch_sizes : int or 1d-array
        - If int, what batch size was used to train the models
        of the specified datatype.
        - If 1d-array, what batch size was used to train the models
        of each of the specified datatypes.
    lrs : float or 1d-array
        - If float, what learning rate was used to train the models
        of the specified datatype.
        - If 1d-array, what learning rate was used to train the models
        of each of the specified datatypes.
    weight_decays : float or 1d-array
        - If float, what weight decay was used to train the models
        of the specified datatype.
        - If 1d-array, what weight decay was used to train the models
        of each of the specified datatypes.
    
    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with the datatypes as index
        and the layer architecture of the best-performing
        architecture for each datatype.
    """

    # Change the type of some of the parameters to an array, if needed.
    if isinstance(datatypes, str):
        datatypes = [datatypes]
    if isinstance(deltas, int):
        deltas = [deltas]
    if isinstance(lengths, int):
        lengths = [lengths]
    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes]
    if isinstance(lrs, float):
        lrs = [lrs]
    if isinstance(weight_decays, float):
        weight_decays = [weight_decays]

    # Initialize output
    layer_params = []
    out = {datatype:{'layers':None} for datatype in datatypes}

    # Iterate through each datatype, get the best layers architecture
    # and add it to the output.
    for i, datatype in enumerate(datatypes):

        # Get best layers and add it to the output.
        best_layers = get_best_layers(
            datatype=datatype, 
            return_layer_params=True, 
            delta=deltas[i], 
            length=lengths[i], 
            horizon=horizon, 
            batch_size=batch_sizes[i], 
            lr=lrs[i], 
            weight_decay=weight_decays[i], 
            output_missing=False)

        out[datatype]['layers'] = best_layers
        layer_params.extend(list(best_layers.keys()))
    
    # Turn the output into a dataframe and add a column
    # for each unique layer parameter.
    df = pd.DataFrame.from_dict(out, orient='index')
    layer_params = sorted(list(set(layer_params)))
    exclude_params = ['conv_paddings', 'dense_hidden_size', #these are all the parameters
                    'dense_activation', 'maxpool_kernels', #that were the same across all
                    'maxpool_paddings', 'maxpool_strides'] #layer architectures     
    layer_params = [i for i in layer_params if not i in exclude_params]
    df[layer_params] = np.nan
    
    # For each layer parameter, save the value of that particular 
    # parameter of the best performing architecture per datatype
    # in the corresponding column.
    for param in layer_params:
        df[param] = df.layers.apply(lambda x: extract_layer_param(x, param))
    
    # Reorder the index to the order of the datatypes input, 
    # drop the redundant 'layers' column, and return the df.
    df = df.reindex(datatypes)
    df.drop(['layers'], axis=1, inplace=True)

    return df
     

# Define a function that compares the mini-batch validation RMSEs
# across different datatypes and forecast horizons.
def compare_best_rmses(datatypes, horizons, batch_sizes, lrs, weight_decays,
                    min_or_mean='min', include_persistence=True, include_epochs=False,
                    include_delta_length=False, include_hyperparams=False):
    """
    Function that returns a dataframe containing the best mini-batch validation
    RMSE for each specified datatype and each forecast horizon. 
    If the hyperparameters are given, then only models are considered
    that were trained with these particular hyperparameters.

    Parameters
    ----------
    datatypes : str or 1d-array
                {'images', 'images3d', 'irradiance', 'weather', 'combined'}
        - If str, shows the best RSME of just a single data type.
        - If 1d-array, shows the best RSME for multiple data types.
    horizons : int or 1d-array
        - If int, dataframe will consist of one column containing
        the best RSME for each datatype for this particular horizon.
        - If 1d-array, dataframe will contain one column for each horizon.
    batch_sizes : None, int, or 1d-array
        - If None, will search the RMSEs of all possible number batch sizes.
        - If int, what batch size was used to train the models
        of the specified datatype.
        - If 1d-array, what batch size was used to train the models
        of each of the specified datatypes.
    lrs : None, float, or 1d-array
        - If None, will search the RMSEs of all possible learning rates.
        - If float, what learning rate was used to train the models
        of the specified datatype.
        - If 1d-array, what learning rate was used to train the models
        of each of the specified datatypes.
    weight_decays : None, float, or 1d-array
        - If None, will search the RMSEs of all possible weight decays.
        - If float, what weight decay was used to train the models
        of the specified datatype.
        - If 1d-array, what weight decay was used to train the models
        of each of the specified datatypes.
    min_or_mean : {'min', 'mean'} (Default: 'min')
        - If min, uses the minimum RMSE per delta/length and horizon.
        - If mean, uses the mean RMSE per delta/length and horizon.
    include_persistence : bool (Default: True)
        - If True, also calculates the RMSE per horizon for the persistence model.
        - If False, does not calculate the RMSE for the persistence model.
    include_epochs : bool (Default: False)
        Whether to also include the number of epochs until convergence as a column
        in the dataframe.
    include_delta_length : bool (Default: False)
        Whether to also include the (delta, length) combination of the input data
        that was used to train the model as a column in the dataframe.
    include_hyperparams : : bool (Default: False)
        Whether to also include the value of the hyperparameters of the model 
        as a column in the dataframe.

    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with the datatypes as index, the specified horizons
        as columns, and the best validation RMSE as values. 
        If specified, also adds other values, e.g. number of epochs
        until convergence per model, to the dataframe.
    """

    # Change the type of some of the parameters to an array, if needed.
    if isinstance(datatypes, str):
        datatypes = [datatypes]
    if isinstance(horizons, int):
        horizons = [horizons]
    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes]
    if isinstance(lrs, float):
        lrs = [lrs]
    if isinstance(weight_decays, float):
        weight_decays = [weight_decays]

    # Define a dictionary of the parameters that we used in the very
    # first training step, i.e. the params used to find the best
    # layer architecture per datatype.
    layer_dict = {'images':{'delta':0,
                            'length':1,
                            'horizon':60,
                            'batch_size':128,
                            'lr':1e-04,
                            'weight_decay':1e-05},
                'images3d':{'delta':10,
                            'length':3,
                            'horizon':60,
                            'batch_size':64, #was needed due to GPU constrains
                            'lr':1e-04,
                            'weight_decay':1e-05},
                'irradiance':{'delta':3, #we used a delta=3 for all LSTMs
                            'length':25, #also same for all LSTMs
                            'horizon':60,
                            'batch_size':128,
                            'lr':1e-04,
                            'weight_decay':1e-05},
                'weather':{'delta':3, 
                            'length':25,
                            'horizon':60,
                            'batch_size':128,
                            'lr':1e-04,
                            'weight_decay':1e-05},
                'combined':{'delta':3, 
                            'length':25,
                            'horizon':60,
                            'batch_size':128,
                            'lr':1e-04,
                            'weight_decay':1e-05}}

    # Initialize output where values are stored.
    out = {datatype:
            {f'y_{horizon}':{'(Delta, Length)':None, 
                            'RMSE':None,
                            'Epochs':None,
                            'Batch Size':None,
                            'Learning Rate':None,
                            'Weight Decay':None}
                for horizon in horizons}
            for datatype in datatypes}
    
    # Iterate through each datatype and get the best performing
    # layer architecture for each.
    for i, datatype in enumerate(datatypes):

        # Get the index of the best performing layer architecture.
        layers_idx = get_best_layers(
            datatype=datatype, 
            return_layer_params=False, 
            **layer_dict[datatype],
            output_missing=False)

        # Define the path where the subsequent RMSEs are stored.
        path =  f'./performance/{datatype}/{datatype}_layers_{layers_idx}'

        # Iterate through all possible deltas for this particular layer index.
        deltas = [d for d in next(os.walk(path))[1]]
        deltas = [i.split('_')[-1] for i in deltas]

        for delta in deltas:

            # Iterate through all possible lengths.
            lengths = [l for l in next(os.walk(f'{path}/delta_{delta}/'))[1]]
            lengths = [i.split('_')[-1] for i in lengths]

            for length in lengths:

                # Iterate through all specified horizons.
                for horizon in horizons:

                    # Get a list of all .pkl files containing the RMSEs
                    rmse_path = f'{path}/delta_{delta}/length_{length}/y_{horizon}'
                    try:
                        rmse_files = [f for f in os.listdir(rmse_path) if not f.startswith('.')
                                        and 'final' not in f] #ensures that the final model trained on
                                                            #100% of the data is excluded here
                    
                    except FileNotFoundError:
                        # This means that this forecast horizon does not exist
                        # for this delta-length combination. 
                        # Thus, we can skip to the next horizon.
                        continue

                    # Get the best mini-batch val. RMSE for each file, check if that RMSE is lower
                    # than the current one in the output, and if it is, store it
                    # in the output.
                    for file in rmse_files:

                        # Extract the hyperparameters from the name.
                        hyperparams = extract_hyperparameters(file.replace('.pkl', ''))

                        # Check if the hyperparameters of this particular file
                        # are among the specified ones in the function parameters.
                        # If None, that means we want to consider all values
                        # for this particular hyperparameter.

                        if batch_sizes is not None:
                            if hyperparams['batch_size'] != str(batch_sizes[i]):
                                continue
                        
                        if lrs is not None:
                            if hyperparams['lr'] != str(lrs[i]):
                                continue

                        if weight_decays is not None:
                            if hyperparams['weight_decay'] != str(weight_decays[i]):
                                continue
                            
                        # If all hyperparameters of this particular file passed
                        # this "filter", then we can read in the df
                        # containing the RMSEs.
                        df = pd.read_pickle(f'{rmse_path}/{file}')
                        
                        # Either get the minimum mini-batch RMSE
                        # or take a mean of the lowest 5 mini-batch RMSEs
                        # to account for fluctuations and non-convergence.
                        if min_or_mean == 'min':
                            rmse = df['val'].min()
                        elif min_or_mean == 'mean':
                            rmse = df['val'].nsmallest(5).mean()

                        # Also extract the number of epochs until convergence.
                        epochs = df.index.max() + 1

                        # Add the rmse, delta/length, number of epochs, 
                        # and hyperparameters to the output, if needed,
                        # i.e. only if the RMSE of the selected model is lower 
                        # than the current benchmark.
                        if out[datatype][f'y_{horizon}']['RMSE'] is None \
                        or out[datatype][f'y_{horizon}']['RMSE'] > rmse:

                            out[datatype][f'y_{horizon}']['RMSE'] = rmse.round(2)
                            out[datatype][f'y_{horizon}']['(Delta, Length)'] = (delta, length)
                            out[datatype][f'y_{horizon}']['Epochs'] = epochs
                            out[datatype][f'y_{horizon}']['Batch Size'] = hyperparams['batch_size']
                            out[datatype][f'y_{horizon}']['Learning Rate'] = hyperparams['lr']
                            out[datatype][f'y_{horizon}']['Weight Decay'] = hyperparams['weight_decay']

    # Prepare output that is shaped for a multi-level column in pandas.
    output = {datatype:{} for datatype in out.keys()}

    # Iterate through each datatype and fill that output.
    for datatype, v in out.items():

        # Extract each horizon and subcolumn.
        for horizon, subcol in v.items():

            # Create a tuple of the horizon and subcol name and 
            # assign it the corresponding value.
            for subcol_name, val in subcol.items():
                output[datatype][(horizon, subcol_name)] = val

    # Create a multi-level column df containing the RMSEs and other values.
    df = pd.DataFrame.from_dict(output, orient='index')

    # Lastly, if specified, calculate the RMSE per horizon of the persistence model.
    if include_persistence:

        print('Making predictions using the persistence model and calculating RMSEs...')
        
        # Initialize the output, the model, and create the correct dataset.
        persistence_rmses = []
        model = models.Persistence()
        dataset = data.create_dataset(
            type='irradiance', from_pkl=True)

        # For each horizon, extract and preprocess the correct data.
        for horizon in horizons:
        
            X, y, t = dataset.get_data(
                datatype='irradiance',
                forecast_horizon=horizon,
                sample=False,
                sample_size=1,
                delta=1,
                length=10)

            (X_train, X_val, X_test,
            y_train, y_val, y_test,
            t_train, t_val, t_test) = preprocessing.split_and_preprocess(
                X=X, y=y, timestamps=t, preprocessing=None)

            # Clean memory as we only need tha validation sets.
            del X, y, t, X_train, y_train, t_train
            del X_test, y_test, t_test
            gc.collect()

            # Get predictions of the model and calculate the errors
            yhat = model(X_val)
            yhat = yhat.detach().numpy().flatten()
            y_val = y_val.detach().numpy().flatten()
            errors = (y_val - yhat)**2

            # To make it comparable, we also need to calculate the RMSE
            # of the persistence model on mini batches.
            rmses = []
            for j in range(0, len(errors), 64): #we'll use a batch size of 64
                rmse_batch = np.sqrt(np.mean(errors[j:j + 64]))
                rmses.append(rmse_batch)
            
            # Append the average mini-batch rmse to the output.
            persistence_rmses.extend([
                (0, 1), #(Delta, Length) of the persistence model
                np.mean(rmses).round(2), #RMSE of persistence
                None, #None for the epoch column as it was not trained
                None, #Same for batch size and the other hyperparams
                None, #learning rate
                None]) #weight decay

        # Add the persistence RMSEs to the df and return it
        df.loc['persistence'] = persistence_rmses

    # Drop some columns depending on which parameters were selected 
    # in the function.
    if include_epochs == False:
        df.drop(columns=['Epochs'], 
            inplace=True, axis=1, level=1)
    if include_delta_length == False:
        df.drop(columns=['(Delta, Length)'], 
            inplace=True, axis=1, level=1)
    if include_hyperparams == False:
        df.drop(columns=['Batch Size', 'Learning Rate', 'Weight Decay'], 
            inplace=True, axis=1, level=1)

    return df


# Define a function that counts all the models that were trained
# by datatype and compares the final RMSE per datatype and horizon.
def compare_final_rmses(datatypes, horizons):
    """
    Function that loops through all possible datatype layer architectures
    and delta-length-horizon combination to count the number of models that
    were trained per datatype. It then aggregates the validation RMSEs and
    its %-difference to the validation mini-batch RMSEs.

    Parameters
    ----------
    datatypes : str or 1d-array
                {'images', 'images3d', 'irradiance', 'weather', 'combined'}
        - If str, shows the final val. RSME of just a single model type.
        - If 1d-array, shows the final val. RSME for multiple models.
    horizons : int or 1d-array
        - If int, dataframe will consist of one column containing
        the best RSME for each datatype for this particular horizon.
        - If 1d-array, dataframe will contain one column for each horizon.
    
    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with the datatypes as index, the specified horizons
        as columns, and the validation RMSE on the whole dataset as well as
        the percent change compared to the mini-batch RMSEs as values. 
        It also includes the total number of models trained per datatype 
        as a last column.
    """

    # Change the type of the parameters to an array, if needed.
    if isinstance(datatypes, str):
        datatypes = [datatypes]
    if isinstance(horizons, int):
        horizons = [horizons]

    # Initialize output where values are stored.
    out = {datatype:
            {f'y_{horizon}':{'RMSE':None,
                            '% Change':None,
                            '# Models':0}
                for horizon in horizons}
            for datatype in datatypes}

    # Iterate through each datatype and each layer architecture.
    # For each, iterate through all possible deltas, lengths, and horizons.
    for datatype in datatypes:

        # Get every directory for each architecture.
        datatype_path = f'./performance/{datatype}'
        layer_dirs = [f'{datatype_path}/{i}' for i in next(os.walk(datatype_path))[1]]

        # Iterate through each layer architecture and get all possible delta directories.
        for layer_path in layer_dirs:

            layer = layer_path.split('_')[-1] #extract the layer value
            delta_dirs = [f'{layer_path}/{i}' for i in next(os.walk(layer_path))[1]]

            # Iterate through all deltas and get all possible length directories.
            for delta_path in delta_dirs:

                delta = delta_path.split('_')[-1] #extract the delta value
                length_dirs = [f'{delta_path}/{i}' for i in next(os.walk(delta_path))[1]]

                # Iterate through all lengths and get all possible horizon directories
                # and the corresponding horizon values.
                for length_path in length_dirs:

                    length = length_path.split('_')[-1] #extract the length value
                    horizon_dirs = [f'{length_path}/{i}' for i in next(os.walk(length_path))[1]]
                    horizon_vals = [int(i.split('y_')[-1]) for i in horizon_dirs]

                    # Iterate though all horizons. Each of these directories will contain
                    # one or multiple .pkl files. Each of these files contains the mini-batch
                    # train and val RMSEs of a unique model that was trained.
                    for (horizon, horizon_path) in zip(horizon_vals, horizon_dirs):

                        # Some of the dataframes in this directory are empty, as the model 
                        # could not be trained. Therefore, we can use the same directory logic
                        # but for the parameters to count the number of trained models, as the
                        # model parameters were only stored if the model could be trained.
                        param_dir = horizon_path.replace('performance', 'parameters')
                        param_files = [f for f in os.listdir(param_dir) if not f.startswith('.')]
                        count = len(param_files)
                        out[datatype][f'y_{horizons[-1]}']['# Models'] += count

                        # If the horizon of the current directory is amomng the ones we want to examine,
                        # then extract the best mini-batch RMSE.
                        if horizon in horizons:

                            # Extract the RMSE df filenames and create the path where they are stored.
                            rmse_files = [f for f in os.listdir(horizon_path) if '.pkl' in f]
                            path =  f'./performance/{datatype}/{datatype}_layers_{layer}'
                            rmse_path = f'{path}/delta_{delta}/length_{length}/y_{horizon}'

                            for file in rmse_files:
                                try:
                                    df = pd.read_pickle(f'{rmse_path}/{file}')
                                    rmse = df['val'].min()
                            
                                    # Add the mini-batch rmse to the output, if needed,
                                    # i.e. only if the RMSE of the selected model is lower 
                                    # than the current benchmark.
                                    if out[datatype][f'y_{horizon}']['% Change'] is None \
                                    or out[datatype][f'y_{horizon}']['% Change'] > rmse:

                                        out[datatype][f'y_{horizon}']['% Change'] = rmse.round(2)

                                except EOFError:
                                    pass

    # We have now added the best mini-batch RMSE per datatype and horizon.
    # We now want the validation RMSE calculated on the whole dataset for each datatype-horizon
    # combination. For this, we can utilize the data.get_ensemble_data() function.
    for horizon in horizons:
        val_df = data.get_ensemble_data(
            horizon=horizon,
            only_return_val_df=True)

        # Calculate the RMSE on the whole dataset for each datatype
        # and add it to the output.
        for datatype in datatypes:

            errors = (val_df[datatype].to_numpy() - val_df['y'].to_numpy())**2
            rmse = np.sqrt(np.mean(errors))
            out[datatype][f'y_{horizon}']['RMSE'] = rmse.round(2)
    
    # We now need to transform the output into a dataframe.
    # Prepare output that is shaped for a multi-level column in pandas.
    output = {datatype:{} for datatype in out.keys()}

    # Iterate through each datatype and fill that output
    for datatype, v in out.items():

        # Extract each horizon and subcolumn
        for horizon, subcol in v.items():

            # Create a tuple of the horizon and subcol name and 
            # assign it the corresponding value.
            for subcol_name, val in subcol.items():
                output[datatype][(horizon, subcol_name)] = val

    # Create a multi-level column df containing the RMSEs and other values.
    df = pd.DataFrame.from_dict(output, orient='index')

    # We also need the RMSEs of the persistence model for each horizon.
    print('Making predictions using the persistence model and calculating RMSEs...')
        
    # Initialize the output, the model, and create the correct dataset.
    persistence_rmses = []
    model = models.Persistence()
    dataset = data.create_dataset(
        type='irradiance', from_pkl=True)

    # For each horizon, extract and preprocess the correct data.
    for horizon in horizons:
    
        X, y, t = dataset.get_data(
            datatype='irradiance',
            forecast_horizon=horizon,
            sample=False,
            sample_size=1,
            delta=1,
            length=10)

        (X_train, X_val, X_test,
        y_train, y_val, y_test,
        t_train, t_val, t_test) = preprocessing.split_and_preprocess(
            X=X, y=y, timestamps=t, preprocessing=None)

        # Clean memory as we only need tha validation sets.
        del X, y, t, X_train, y_train, t_train
        del X_test, y_test, t_test
        gc.collect()

        # Get predictions of the model and calculate the errors.
        yhat = model(X_val)
        yhat = yhat.detach().numpy().flatten()
        y_val = y_val.detach().numpy().flatten()
        errors = (y_val - yhat)**2
        rmse = np.sqrt(errors.mean()).round(2)

        # We also need to calculate the RMSE of the persistence model
        # but this time using mini-batches.
        mini_rmses = []
        for j in range(0, len(errors), 64): #we'll use a batch size of 64
            rmse_batch = np.sqrt(np.mean(errors[j:j + 64]))
            mini_rmses.append(rmse_batch)
        mini_rmse = np.mean(mini_rmses).round(2)

        # Append the rmse to the output.
        persistence_rmses.extend([
            rmse, #RMSE of persistence
            mini_rmse, #RMSE of persistence using mini-batches
            None]) #None for the '# Models' column
           
    # Add the persistence RMSEs to the df and return it
    df.loc['persistence'] = persistence_rmses
    
    # We can drop all but the last '# Models' column as this is where we aggregated 
    # the count of all models. We did this, as this count includes every model
    # across ALL layer architectures, deltas, lengths, horizons,
    # and hyperparams. Hence, we did not need to split it by horizon.
    if len(horizons) > 1:

        for horizon in horizons[:-1]:

            df.drop((f'y_{horizon}', '# Models'), axis=1, inplace=True)

    # Lastly we need to calculate the percentage change between the RMSE
    # calculated during training vs. the RMSE on the whole data.
    for horizon in horizons:

        # Calculate the percentage change.
        # Then, format it correctly.
        df.loc[:, (f'y_{horizon}', '% Change')] = (df.loc[:, (f'y_{horizon}', 'RMSE')] - \
                                                    df.loc[:, (f'y_{horizon}', '% Change')]) / \
                                                    df.loc[:, (f'y_{horizon}', '% Change')]
        df.loc[:, (f'y_{horizon}', '% Change')] = (df.loc[:, (f'y_{horizon}', '% Change')] * 100).round(2)

    return df


# Define a function that fits a linear regression across different
# ensemble data combinations and forecast horizons. It then aggregates
# each model's train and validation performance into a dataframe.
def compare_linear_combinations(horizons, include_test_errors=False):
    """
    Function that first fits a linear regression onto the data
    generated by data.generate_ensemble_data(). It repeats this step
    for all possible combinations of the different parameters that
    generate_ensemble_data() uses, e.g. include_time=True and
    include_time=False. It then repeats this for all given forecast
    horizons.

    Parameters
    ----------
    horizons : int or 1d-array
        - If int, dataframe will consist of one column containing
        the best train and val RSME for this particular horizon.
        - If 1d-array, dataframe will contain one column for each horizon.
    include_test_errors : bool (Default: False)
        - If True, also includes the test error of each linear combination.
        - If False, only includes train and val, but not test errors.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe with one row per ensemble input combination and 
        one column per horizon containing train and val RMSEs of the
        linear ensemble baseline.
    """

    # Change the type of horizons to an array, if needed.
    if isinstance(horizons, int):
        horizons = [horizons]

    # Denote the different datatypes that can be generated from data.get_ensemble_data().
    # Then add all possible data combinations to a list.
    ensemble_datatypes = ['interactions', 'time', 'hour_poly', 'ghi', 'weather']
    combos = [comb for i in range(len(ensemble_datatypes)) 
                for comb in itertools.combinations(ensemble_datatypes, i + 1)]
    combos = [()] + combos
    
    # We can now prepare the output that stores the train and val RMSE for each
    # combo and horizon.
    if include_test_errors == False:
        out = {'predictions+' + '+'.join(comb): #all data combinations incl. predictions
                {f'y_{horizon}': #all forecast horizons
                    {'Val RMSE':None, 'Train RMSE':None}
                    for horizon in horizons}
                for comb in combos} #both RMSEs
    
    # Modify this output if we also want to store the test errors
    elif include_test_errors == True:
        out = {'predictions+' + '+'.join(comb): 
                {f'y_{horizon}': 
                    {'Val RMSE':None, 'Train RMSE':None, 'Test RMSE':None}
                    for horizon in horizons}
                for comb in combos}

    # We can now iterate through each different data combination and forecast horizon. 
    # For each combo, we can create a dictionary containing the content of the combo 
    # as key and a boolean True as value, e.g. {'include_interactions':True}.
    for horizon in horizons:

        print(f'Fitting linear combinations on ensemble data with horizon: {horizon}')

        for comb in combos:

            # Create the boolean dictionary.
            comb_dict = {f'include_{key}':True for key in comb}

            # We can then use this dictionary to generate the ensemble data.
            X_train, X_val, X_test, \
                y_train, y_val, y_test, \
                t_train, t_val, t_test, \
                ensemble_labels = data.get_ensemble_data(
                    horizon=horizon,
                    **comb_dict)

            # Now fit a linear regression to the train data.
            model = LinearRegression()
            model.fit(X_train ,y_train)

            # Calculate the RMSE on the train data.
            errors = y_train - model.predict(X_train)
            errors = errors ** 2
            rmse_train = np.sqrt(np.mean(errors)).round(2)

            # And on the validation data.
            errors = y_val - model.predict(X_val)
            errors = errors ** 2
            rmse_val = np.sqrt(np.mean(errors)).round(2)

            # Add both RMSEs to the output for the given data combo and horizon.
            key = 'predictions+' + '+'.join(comb)
            out[key][f'y_{horizon}']['Train RMSE'] = rmse_train
            out[key][f'y_{horizon}']['Val RMSE'] = rmse_val
            
            # If specified, also repeat this for the test errors.
            if include_test_errors == True:
                errors = y_test - model.predict(X_test)
                errors = errors ** 2
                rmse_test = np.sqrt(np.mean(errors)).round(2)
                out[key][f'y_{horizon}']['Test RMSE'] = rmse_test

    # Prepare output that is shaped for a multi-level column in pandas.
    output = {comb:{} for comb in out.keys()}

    # Iterate through each comb and fill that output.
    for comb, v in out.items():

        # Extract each horizon and subcolumn.
        for horizon, subcol in v.items():

            # Create a tuple of the horizon and subcol name and 
            # assign it the corresponding value.
            for subcol_name, val in subcol.items():
                output[comb][(horizon, subcol_name)] = val

    # Create a multi-level column df containing the RMSEs and other values.
    df = pd.DataFrame.from_dict(output, orient='index')

    return df.round(2)


# Define a function that compares the linear coefficients across horizons.
def compare_linear_coef(horizons, **kwargs):
    """
    Function that fits a linear combination model for each horizon on the
    specified ensemble data combinations (e.g. with include_time=True).
    It then returns a dataframe showing the weight of each coefficient
    of this linear model for each forecast horizon.

    Parameters
    ----------
    horizons : int or 1d-array
        - If int, dataframe will consist of one column containing
        the coefficients of the linear model for this particular horizon.
        - If 1d-array, dataframe will contain one column for each horizon.
    **kwargs : {include_interactions, include_time, include_ghi, include_weather}
        Parameters of the function data.get_ensemble_data() that denote which
        ensemble data combinations the models should be trained on,
        e.g. include_time=True.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe consisting of the coefficients and their weights of the
        linear model trained on the ensemble data per forecast horizon.
    """

    # Change the type of horizons to an array, if needed.
    if isinstance(horizons, int):
        horizons = [horizons]

    # Create the variable that will contain our final dataframe.
    coef_df = None

    # Iterate through each horizon and fit the linear regression
    # on the specified ensemble data.
    for horizon in horizons:

        X_train, _, _, y_train, _, _, _, _, _, ensemble_labels = data.get_ensemble_data(
            horizon=horizon,
            **kwargs) #e.g. include_time=True

        # Fit the model and extract the coefficients.
        model = LinearRegression()
        model.fit(X_train ,y_train)
        coef = model.coef_

        # If this is the first horizon in the list, create the dataframe.
        if coef_df is None:

            coef_df = pd.DataFrame(
                data=coef,
                index=ensemble_labels,
                columns=[f'y_{horizon}'])

        # Otherwise, just add the coefficients to the df.
        else:
            
            coef_df[f'y_{horizon}'] = coef

    return coef_df.round(2)
    

# Define a function that finds the best performing ensemble layer architecture.
def get_best_ensemble_layers(batch_size, lr, weight_decay, return_idx=False):
    """
    Function that searches across all different ensemble model layer architectures
    trained on a forecast horizon of y_120 and with the individual model predictions
    as input. For each trained model it calculates the full validation data RMSE
    and then returns the layer architecture of the best model.
    
    Parameters
    ----------
    batch_size : int
        What batch sizes were used to train the models.
    lr : float
        What learning rates were used to train the models.
    weight_decay : float
        What weight decay was used to train the models.
    return_idx : bool (Default: False)
        - If True, returns the index of the layer architecture.
        - If False, returns the parameters of the layer architecture.

    Returns
    -------
    best_layers : dict
        Dictionary of best parameters used to construct the models.NN() class.
        Only if if return_idx=False.
    best_idx : int
        Returns the integer of the best-performing layer index.
        Only if return_idx=True.
    """

    # Initialize the best index, layer architecture, and RMSE.
    best_idx = None
    best_layers = None
    best_rmse = None

    # Generate the ensemble input data.
    _, X_val, _, _, y_val, _, _, _, _, _ = data.get_ensemble_data(
        horizon=120,
        include_interactions=False,
        include_time=False,
        include_ghi=False,
        include_weather=False)

    # Generate the filename based on given hyperparameters.
    filename = f'batch_size({batch_size})__lr({lr})__weight_decay({weight_decay})' \
        '__early_stopping(True).pt'

    # Read the dataframe containing all ensemble layer architectures.
    layer_df = pd.read_pickle('./parameters/ensemble/ensemble_layers.pkl')

    # Iterate through all ensemble layer architectures and extract the layer index.
    # and the path where the trained model is stored.
    filepath = './parameters/ensemble/'
    ensemble_layers = [f for f in os.listdir(filepath) if '.' not in f]
    for layer in ensemble_layers:

        idx = layer.split('_')[-1]
        model_path = f'{filepath}{layer}/predictions__/y_120/{filename}'

        # Extract the correct layer architecture and build that model.
        layer_params = layer_df.loc[int(idx), 'ensemble_layers']
        model = models.NN(
            input_dim=X_val.shape[1],
            **layer_params)
        
        # Load in the trained model parameters for this layer index.
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            model = model.eval()

            # Calculate the full validation RMSE.
            y_hat = model(torch.from_numpy(X_val))
            y_hat = np.array(y_hat.squeeze(1).detach())
            errors = (y_hat - y_val) ** 2
            rmse = np.sqrt(np.mean(errors))

            # If this is currently the best RMSE, save this index and layer params.
            if best_rmse is None:
                best_idx = idx
                best_layers = layer_params
                best_rmse = rmse

            elif rmse < best_rmse:
                best_idx = idx
                best_layers = layer_params
                best_rmse = rmse

        except FileNotFoundError:
            pass

    if return_idx == False:
        return best_layers
    elif return_idx == True:
        return best_idx


# Define a function that compares the performances of the the linear 
# vs NN ensemble models across data input combinations and forecast horizons.
def compare_linear_vs_nn(horizons, batch_size, lr, weight_decay):
    """
    Function that first fits the linear models on the ensemble data combinations.
    It then searches for the trained NN model parameters for each data combination
    and forecast horizon and calculates its full validation RMSE. This is then
    compared (in %) to the RMSE of the linear model fitted on the respective data
    and forecast horizon.

    Parameters
    ----------
    horizons : int or 1d-array
        - If int, dataframe will consist of one column containing
        the best NN validation RMSE and its comparison the linear model.
        - If 1d-array, dataframe will contain one column for each horizon.
    batch_size : 'best' or int
        - If 'best', chooses the best performing batch size per NN model
        and per horizon.
        - If int, uses the integer value to search for the trained model
        across all horizons.
    lr : 'best' or float
        - If 'best', chooses the best performing learning rate per NN model
        and per horizon.
        - If float, uses the float value to search for the trained model
        across all horizons.
    weight_decay : 'best' or float
        - If 'best', chooses the best performing weight decay per NN model
        and per horizon.
        - If float, uses the float value to search for the trained model
        across all horizons.

    Returns
    -------
    df : pd.DataFrame
        Dataframe containing the full validation RMSE of the NN ensemble
        models for each ensemble input combination. This is compared (in %)
        against the respective linear baseline for all given horizons.
    """

    # Change the type of horizons to an array, if needed.
    if isinstance(horizons, int):
        horizons = [horizons]
    
    # Fit the linear ensemble model on each given horizon and
    # get their RMSEs.
    df = compare_linear_combinations(horizons)

    # Get the best performing NN ensemble model layer architecture index.
    best_idx = get_best_ensemble_layers(
        batch_size=128, #we used these hyperparams to find the best layers
        lr=0.0001,
        weight_decay=1e-05,
        return_idx=True)

    # Also get the best performing layer architecture parameters.
    best_layers = get_best_ensemble_layers(
        batch_size=128, #we used these hyperparams to find the best layers
        lr=0.0001,
        weight_decay=1e-05)

    # Create the filename based on the given hyperparameters.
    filename = f'batch_size({batch_size})__lr({lr})__weight_decay({weight_decay})' \
        '__early_stopping(True).pt'

    # Iterate through each horizon change the df to prepare the comparison
    # of NN ensemble vs linear ensemble model.
    for horizon in horizons:

        # Remove the train RMSEs contents.
        df.loc[:, (f'y_{horizon}', 'Train RMSE')] = df.loc[:, (f'y_{horizon}', 'Val RMSE')]        

        # Prepare the list that stores the full validation RMSEs of the NN ensemble models.
        # Then find the NN ensemble model trained on each ensemble data combination,
        # and calculate its full validation RMSE.
        nn_rmses = [np.nan for i in range(len(df))]

        print('Calculating the full validation RMSE with the NN ensemble model '
                f'for every ensemble input combination with forecast horizon: {horizon}')
        
        # Iterate through each data combination.
        for i, data_comb in enumerate(df.index):

            comb = data_comb.replace('+', '__') #this is how the folders are named
            path = f'./parameters/ensemble/ensemble_layers_{best_idx}/{comb}/' \
                f'y_{horizon}/{filename}'

            # Check which type of ensemble data should be generated.
            which_input = models.extract_ensemble_inputs(comb)

            # Generate that particular ensemble data.
            _, X_val, _, _, y_val, _, _, _, _, _ = data.get_ensemble_data(
                horizon=horizon,
                **which_input) #e.g. include_time=True
            
            # Construct the model based on the shape of the input 
            # and using the best performing layer architecture.
            model = models.NN(
                input_dim=X_val.shape[1],
                **best_layers)
        
            # Load in the trained model parameters for this layer index.
            try:
                checkpoint = torch.load(path, map_location='cpu')
                model.load_state_dict(checkpoint['state_dict'])
                model = model.eval()

                # Calculate the full validation RMSE and add it to our list.
                y_hat = model(torch.from_numpy(X_val).float())
                y_hat = np.array(y_hat.squeeze(1).detach())
                errors = (y_hat - y_val) ** 2
                rmse = np.sqrt(np.mean(errors))
                nn_rmses[i] = rmse

            except FileNotFoundError:
                pass
        
        # Add the RMSEs of the NN ensemble to the df and change the columns.
        df.loc[:, (f'y_{horizon}', 'Val RMSE')] = nn_rmses

        # Calculate the % between the linear ensemble model and the
        # NN ensemble model.
        df.loc[:, (f'y_{horizon}', 'Train RMSE')] = \
            (df.loc[:, (f'y_{horizon}', 'Val RMSE')] \
            - df.loc[:, (f'y_{horizon}', 'Train RMSE')]) \
            / df.loc[:, (f'y_{horizon}', 'Train RMSE')]
        df.loc[:, (f'y_{horizon}', 'Train RMSE')] = (df.loc[:, (f'y_{horizon}', 'Train RMSE')]*100).round(2)

    # Rename columns (the Train RMSE column is now the % between NN and linear)
    df.columns.set_levels(['% vs. linear', 'Val RMSE'],level=1,inplace=True)
    
    return df.round(2)


# Define a function that returns the model and input parameters of the 
# best trained ensemble model for a given forecast horizon.
def get_best_ensemble_model_by_horizon(horizon, return_inputs, return_hyperparams,
                                batch_size, lr, weight_decay):
    """
    Function that returns the best model and all the necessary parameters,
    i.e. layer parameters, inputs, and hyperparams,
    to load in the best NN ensemble model for a given forecast horizon.

    Parameters
    ----------
    horizon : int
        What forecast horizon the model was trained on.
    return_inputs : bool
        Whether to also a return a dictionary that can be unpacked unpacked in 
        data.get_ensemble_data() to generate the best performing ensemble inputs.
    return_hyperparams : bool
        Whether to also return a dictionary containing the values of the
        hyperparameters of the the model that is returned.
    batch_size : None or int
        - If None, will compare the RMSEs of all possible batch sizes.
        - If int, will compare the RMSEs of this specified batch size.
    lr : None or float
        - If None, will compare the RMSEs of all possible learning rates.
        - If float, will compare the RMSEs of this specified learning rates.
    weight_decay : None or float
        - If None, will compare the RMSEs of all possible weight decays
        - If float, will compare the RMSEs of this specified weight decay.
    
    Returns
    -------
    best_model : torch.nn.Module
        Loaded PyTorch model that has the best full validation RMSE
        across all layer architectures, ensemble inputs, and 
        hyperparameters for a given forecast horizon.
    best_inputs : dict
        Dictionary that can be unpacked in data.get_ensemble_data() to generate 
        the best performing ensemble inputs used to train the model.
        Only if return_inputs=True.
    best_hyperparams : dict
        Also returns a dictionary of hyperparameters used to train the model.
        Only if return_hyperparams=True.
    """

    # We have trained the models on three forecast horizon buckets:
    # intra-hour, intra-day, and day-ahead. We used a horizon of
    # 30min, 2h, and 1d for each of these buckets respectively.
    # Thus, if this function gets a horizon that is not one of these
    # three, we need to change it to the correct value of the bucket.
    if horizon < 60:
        horizon = 30 #change all such horizons to the 30min used for training
                        #of the intra-hour models
    elif horizon < 60*24:
        horizon = 120 #change all such horizons to the 2h used for training
                        #of the intra-day models
    else:
        horizon = 1440 #change all such horizons to the 1d used for training
                        #of the day-ahead models

    # Get the best performing layer index of the ensemble model.
    idx = get_best_ensemble_layers(
        batch_size=128, #we used these hyperparams to optimize the architecture
        lr=1e-04,
        weight_decay=1e-05,
        return_idx=True)

    # Also get the best performing layer architecture parameters.
    best_layers = get_best_ensemble_layers(
        batch_size=128, #we used these hyperparams to find the best layers
        lr=1e-04,
        weight_decay=1e-05)

    # Initialize the parameters that we want to store
    best_inputs = None
    best_hyperparams = None
    best_rmse = None
    best_model = None

    print('Calculating the full validation RMSE for every ensemble '
        f'input combination w/ the NN ensemble model for horizon y_{horizon}. ' 
        'This can take a few minutes.')

    # Iterate through all ensemble data combinations
    path = f'./parameters/ensemble/ensemble_layers_{idx}/'
    combs = [f for f in os.listdir(path) if '.' not in f] #only relevant folders
    for comb in combs:

        # Generate the ensemble data for each ensemble data combination.
        which_inputs = models.extract_ensemble_inputs(comb)
        _, X_val, _, _, y_val, _, _, _, _, _ = data.get_ensemble_data(
            horizon=horizon,
            **which_inputs) #e.g. include_time=True

        # Construct the model based on the shape of the input 
        # and using the best performing layer architecture.
        model = models.NN(
            input_dim=X_val.shape[1],
            **best_layers)

        # Iterate through all filenames in the folder of this 
        # ensemble input combination and the specified horizon.
        model_dir = f'{path}{comb}/y_{horizon}/'
        filenames = [f for f in os.listdir(model_dir) if not f.startswith('.')]
        for filename in filenames:

            # Extract the hyperparameters from the filename.
            hyper_string = filename.replace('__early_stopping(True).pt', '')
            hyper_string = hyper_string.replace('__early_stopping(False).pt', '')
            hyperparams = extract_hyperparameters(
                hyperparam_name=hyper_string,
                as_string=False)

            # Check if the hyperparameters of this particular file
            # are among the specified ones in the function parameters.
            # If None, that means we want to consider all values
            # for this particular hyperparameter.

            if batch_size is not None:
                if hyperparams['batch_size'] != batch_size:
                    continue
            
            if lr is not None:
                if hyperparams['lr'] != lr:
                    continue

            if weight_decay is not None:
                if hyperparams['weight_decay'] != weight_decay:
                    continue
                
            # If all hyperparameters of this particular file passed
            # this "filter", we can load in the respective model 
            # with this architecture, input, and hyperparam
            # and calculate the full validation RMSE.
            try:
                model_path = model_dir + filename
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['state_dict'])
                model = model.eval()

                # Calculate the full validation RMSE.
                y_hat = model(torch.from_numpy(X_val).float())
                y_hat = np.array(y_hat.squeeze(1).detach())
                errors = (y_hat - y_val) ** 2
                rmse = np.sqrt(np.mean(errors))
                
                # If this is currently the best RMSE, save variables that
                # we want to output later.
                if best_rmse is None:
                    best_model = model
                    best_inputs = which_inputs
                    best_hyperparams = hyperparams
                    best_rmse = rmse

                elif rmse < best_rmse:
                    best_model = model
                    best_inputs = which_inputs
                    best_hyperparams = hyperparams
                    best_rmse = rmse

            except FileNotFoundError:
                pass

    # Return the specified output(s).
    if return_inputs == False and return_hyperparams == False:
        return best_model
    
    elif return_inputs == True and return_hyperparams == False:
        return best_model, best_inputs

    elif return_inputs == False and return_hyperparams == True:
        return best_model, best_hyperparams

    elif return_inputs == True and return_hyperparams == True:
        return best_model, best_inputs, best_hyperparams


# Define a function that aggregates the min, mean, and SD of the
# mini-batch validation RMSEs of the individual models across all layer architectures.
def get_indiv_performance_of_layers():
    """
    Function that iterates through all five datatypes
    and all their respective layer architecture to aggregate their
    mini-batch validation RMSEs. It then outputs the min., mean,
    and SD of these RMSEs per datatype.

    Parameters
    ----------
    None

    Returns
    -------
    df_indiv_perf_layer : pd.DataFrame
        Pandas dataframe w/ one row per datatype and four columns
        for the min, mean, and SD mini-batch validation RMSEs
        as well as the number of models.
    """

    # Define the initial params that were used to train the different architectures.
    param_dict = {
        'images':{
            'delta':0, 'length':1, 'y':60, #all were trained on a 1-hour ahead forecast
            'batch_size':128, 'lr':0.0001, 'weight_decay':1e-05},
        'images3d':{
            'delta':10, 'length':3, 'y':60, 
            'batch_size':64, 'lr':0.0001, 'weight_decay':1e-05},
        'irradiance':{
            'delta':3, 'length':25, 'y':60, 
            'batch_size':128, 'lr':0.0001, 'weight_decay':1e-05},
        'weather':{
            'delta':3, 'length':25, 'y':60, 
            'batch_size':128, 'lr':0.0001, 'weight_decay':1e-05},
        'combined':{
            'delta':3, 'length':25, 'y':60, 
            'batch_size':128, 'lr':0.0001, 'weight_decay':1e-05}}

    # Initialize the output to save the mini-batch RMSEs and the resulting metrics.
    datatypes = ['images', 'images3d', 'irradiance', 'weather', 'combined']
    out = {datatype:[] for datatype in datatypes}
    metric_out = {datatype:{'min':None, 'mean':None, 'SD':None, 'n':None} 
        for datatype in datatypes}

    # Iterate through each datatype.
    perf_dir = './performance'
    for datatype in datatypes:

        # Extract the correct delta, length, y, and hyperparameter values.
        delta = param_dict[datatype]["delta"]
        length = param_dict[datatype]["length"]
        y = param_dict[datatype]["y"]
        batch_size = param_dict[datatype]["batch_size"]
        lr = param_dict[datatype]["lr"]
        weight_decay = param_dict[datatype]["weight_decay"]

        # Extract the layer indices to iterate through their directories.
        layer_dir = f'{perf_dir}/{datatype}'
        layers = [f for f in os.listdir(layer_dir) if '.' not in f]
        for layer in layers:

            # Go into each layer dir and read in the performance df based on the
            # correct delta, length, y, and hyperparameter values.
            df_file = f'batch_size({batch_size})__lr({lr})__weight_decay({weight_decay}).pkl'
            df_path = f'{layer_dir}/{layer}/delta_{delta}/length_{length}/y_{y}/{df_file}'
            df = pd.read_pickle(df_path)

            # Extract the best mini-batch validation RMSE. However, as the early-stopping
            # rule has not been implemented in this step, use the mean of the min. five
            # mini-batch validation RMSEs. Then, add it to our output.
            rmse = rmse = df['val'].nsmallest(5).mean()
            out[datatype].append(rmse)

        # Calculate the min, mean, and SD of these aggregated mini-batch validation RMSEs.
        metric_out[datatype]['min'] = min(out[datatype]).round(2)
        metric_out[datatype]['mean'] = np.mean(out[datatype]).round(2)
        metric_out[datatype]['SD'] = np.std(out[datatype]).round(2)
        metric_out[datatype]['n'] = len(out[datatype])

    # Aggregate the metrics into a pandas dataframe and return it
    df_indiv_perf_layer = pd.DataFrame.from_dict(metric_out, orient='index')
    
    return df_indiv_perf_layer


# Define a function that aggregates the min, mean, and SD of the
# mini-batch validation RMSEs of the individual models across all delta-length inputs.
def get_indiv_performance_of_inputs(horizon):
    """
    Function that iterates through all five datatypes
    and all their delta-length input combination for their fixed
    best performing layer architecture. It then aggregates their
    mini-batch validation RMSEs and outputs the min., mean,
    and SD of these RMSEs per datatype.

    Parameters
    ----------
    horizon : int
        Which horizon the models were trained on.

    Returns
    -------
    df_indiv_perf_inputs : pd.DataFrame
        Pandas dataframe w/ one row per datatype and four columns
        for the min, mean, and SD mini-batch validation RMSEs
        as well as the number of models.
    """

    # Define the initial params that were used to train the different architectures
    # and denote the lr used for this training step, i.e. to train across delta-lengths.
    param_dict = {
        'images':{
            'delta':0, 'length':1, 'y':60, #all were trained on a 1-hour ahead forecast
            'batch_size':128, 'lr':0.0001, 'weight_decay':1e-05,
            'lr_inputs':0.0001}, #the same lr was used as in the first train step
        'images3d':{
            'delta':10, 'length':3, 'y':60, 
            'batch_size':64, 'lr':0.0001, 'weight_decay':1e-05,
            'lr_inputs':0.0001}, #the same lr was used as in the first train step
        'irradiance':{
            'delta':3, 'length':25, 'y':60, 
            'batch_size':128, 'lr':0.0001, 'weight_decay':1e-05,
            'lr_inputs':3e-04}, #the lr of the LSTMs was changed vs. the first train step
        'weather':{
            'delta':3, 'length':25, 'y':60, 
            'batch_size':128, 'lr':0.0001, 'weight_decay':1e-05,
            'lr_inputs':3e-04}, #the lr of the LSTMs was changed vs. the first train step
        'combined':{
            'delta':3, 'length':25, 'y':60, 
            'batch_size':128, 'lr':0.0001, 'weight_decay':1e-05,
            'lr_inputs':3e-04}} #the lr of the LSTMs was changed vs. the first train step

    # Initialize the output to save the mini-batch RMSEs and the resulting metrics.
    datatypes = ['images', 'images3d', 'irradiance', 'weather', 'combined']
    out = {datatype:[] for datatype in datatypes}
    metric_out = {datatype:{'min':None, 'mean':None, 'SD':None, 'n':None} 
        for datatype in datatypes}

    # Iterate through each datatype.
    perf_dir = './performance'
    for datatype in datatypes:

        # Extract the hyperparameter values used to train across delta-lengths.
        batch_size = param_dict[datatype]["batch_size"]
        lr = param_dict[datatype]["lr_inputs"] #lr used in this training step
        weight_decay = param_dict[datatype]["weight_decay"]

        # Get the best performing layer architecture.
        best_layer_idx = get_best_layers(
            datatype=datatype,
            return_layer_params=False,
            delta=param_dict[datatype]['delta'], 
            length=param_dict[datatype]['length'], 
            horizon=param_dict[datatype]['y'],
            batch_size=param_dict[datatype]['batch_size'],
            lr=param_dict[datatype]['lr'],
            weight_decay=param_dict[datatype]['weight_decay'],
            output_missing=False)

        # Enter the directory of this best architecture, then iterate through
        # all trained deltas.
        delta_dir = f'{perf_dir}/{datatype}/{datatype}_layers_{best_layer_idx}'
        deltas = [f for f in os.listdir(delta_dir) if not f.startswith('.')]

        for delta in deltas:

            # Iterate through all trained lengths.
            length_dir = f'{delta_dir}/{delta}'
            lengths = [f for f in os.listdir(length_dir) if not f.startswith('.')]

            for length in lengths:

                # Enter the specified horizon and extract the correct file based on 
                # the hyperparams used in this training step.
                df_file = f'batch_size({batch_size})__lr({lr})__weight_decay({weight_decay}).pkl'
                df_path = f'{length_dir}/{length}/y_{horizon}/{df_file}'
                df = pd.read_pickle(df_path)

                # Extract the best mini-batch validation RMSE. As from this training
                # step onwards, the early-stopping rule has been implemented,
                # we can just use the min. mini-batch validation RMSE. 
                # Then, add it to our output.
                rmse = rmse = df['val'].min()
                out[datatype].append(rmse)

        # Calculate the min, mean, and SD of these aggregated mini-batch validation RMSEs.
        metric_out[datatype]['min'] = min(out[datatype]).round(2)
        metric_out[datatype]['mean'] = np.mean(out[datatype]).round(2)
        metric_out[datatype]['SD'] = np.std(out[datatype]).round(2)
        metric_out[datatype]['n'] = len(out[datatype])

    # Aggregate the metrics into a pandas dataframe and return it
    df_indiv_perf_inputs = pd.DataFrame.from_dict(metric_out, orient='index')
    
    return df_indiv_perf_inputs


# Define a function that aggregates the min, mean, and SD of the
# mini-batch validation RMSEs of the individual models across all hyperparameters.
def get_indiv_performance_of_hyperparams(horizon):
    """
    Function that iterates through all five datatypes
    and all their hyperparameter combination for their fixed
    best performing layer architecture and best performing
    delta-length inputs. It then aggregates their
    mini-batch validation RMSEs and outputs the min., mean,
    and SD of these RMSEs per datatype.

    Parameters
    ----------
    horizon : int
        Which horizon the models were trained on.

    Returns
    -------
    df_indiv_perf_hyperparams : pd.DataFrame
        Pandas dataframe w/ one row per datatype and four columns
        for the min, mean, and SD mini-batch validation RMSEs
        as well as the number of models.
    """

    # Define the initial params that were used to train the different architectures
    # and denote the lr used to train on the different delta-length inputs.
    param_dict = {
        'images':{
            'delta':0, 'length':1, 'y':60, #all were trained on a 1-hour ahead forecast
            'batch_size':128, 'lr':0.0001, 'weight_decay':1e-05,
            'lr_inputs':0.0001}, #the same lr was used as in the first train step
        'images3d':{
            'delta':10, 'length':3, 'y':60, 
            'batch_size':64, 'lr':0.0001, 'weight_decay':1e-05,
            'lr_inputs':0.0001}, #the same lr was used as in the first train step
        'irradiance':{
            'delta':3, 'length':25, 'y':60, 
            'batch_size':128, 'lr':0.0001, 'weight_decay':1e-05,
            'lr_inputs':3e-04}, #the lr of the LSTMs was changed to train on diff. inputs
        'weather':{
            'delta':3, 'length':25, 'y':60, 
            'batch_size':128, 'lr':0.0001, 'weight_decay':1e-05,
            'lr_inputs':3e-04}, #the lr of the LSTMs was changed to train on diff. inputs
        'combined':{
            'delta':3, 'length':25, 'y':60, 
            'batch_size':128, 'lr':0.0001, 'weight_decay':1e-05,
            'lr_inputs':3e-04}} #the lr of the LSTMs was changed to train on diff. inputs

    # Initialize the output to save the mini-batch RMSEs and the resulting metrics.
    datatypes = ['images', 'images3d', 'irradiance', 'weather', 'combined']
    out = {datatype:[] for datatype in datatypes}
    metric_out = {datatype:{'min':None, 'mean':None, 'SD':None, 'n':None} 
        for datatype in datatypes}

    # Iterate through each datatype.
    perf_dir = './performance'
    for datatype in datatypes:

        # Get the best performing delta and length for the specified horizon.
        _, delta, length = get_best_model_by_horizon(
            datatype=datatype, 
            horizon=horizon, 
            return_delta_length=True, 
            return_hyperparams=False,
            batch_size=param_dict[datatype]['batch_size'], 
            lr=param_dict[datatype]['lr_inputs'], 
            weight_decay=param_dict[datatype]['weight_decay'])

        # Get the best performing layer architecture.
        best_layer_idx = get_best_layers(
            datatype=datatype,
            return_layer_params=False,
            delta=param_dict[datatype]['delta'], 
            length=param_dict[datatype]['length'], 
            horizon=param_dict[datatype]['y'],
            batch_size=param_dict[datatype]['batch_size'],
            lr=param_dict[datatype]['lr'],
            weight_decay=param_dict[datatype]['weight_decay'],
            output_missing=False)

        # Get the directory containing the performances of each different
        # hyperparameter combination.
        hyper_dir = f'{perf_dir}/{datatype}/{datatype}_layers_{best_layer_idx}/' \
                    f'delta_{delta}/length_{length}/y_{horizon}'
        hyperparams = [f for f in os.listdir(hyper_dir) if not f.startswith('.')]

        # Iterate through each hyperparameter combinatin and extract the
        # mini-batch validation RMSE.
        for hyperparam in hyperparams:

            # Extract the hyperparameters from the filename.
            hyperparam_name = hyperparam.replace('.pkl', '')
            hyper_dict = extract_hyperparameters(
                hyperparam_name=hyperparam_name,
                as_string=False)

            # Skip this particular pickle file if it contains the mini-batch RMSEs
            # of an LSTM model trained using a learning rate that was used to train
            # the models across delta-lengths but not when tuning the hyperparameters.
            # This ensures that these models are not double counted across 
            # get_indiv_performance_of_inputs vs. get_indiv_performance_of_hyperparams.
            
            if datatype in ['irradiance', 'weather', 'combined']: #all LSTMs
                if hyper_dict['lr'] == param_dict[datatype]['lr_inputs']: #ignore this lr
                    continue

            # If the lr is different to this previous used lr, continue aggregating
            # mini-batch validation RMSEs. For this, read in the performance df
            # extract the best RMSE and add it to the output
            df_file = f'{hyper_dir}/{hyperparam}'
            df = pd.read_pickle(df_file)
            rmse = rmse = df['val'].min()
            out[datatype].append(rmse)

        # Calculate the min, mean, and SD of these aggregated mini-batch validation RMSEs.
        metric_out[datatype]['min'] = min(out[datatype]).round(2)
        metric_out[datatype]['mean'] = np.mean(out[datatype]).round(2)
        metric_out[datatype]['SD'] = np.std(out[datatype]).round(2)
        metric_out[datatype]['n'] = len(out[datatype])

    # Aggregate the metrics into a pandas dataframe and return it
    df_indiv_perf_hyperparams = pd.DataFrame.from_dict(metric_out, orient='index')
    
    return df_indiv_perf_hyperparams

      
# Define a function that aggregates the min, mean, and SD of the full
# validation RMSEs of the ensemble neural networks across all layer architectures.
def get_ensemble_performance_of_layers():
    """
    Function that calculates the full validation RMSE of every
    layer architecture of the neural network ensemble model. 
    It then outputs the min., mean, and SD of these RMSEs.

    Parameters
    ----------
    None

    Returns
    -------
    df_ensemble_perf_layer : pd.DataFrame
        Pandas dataframe w/ one row and three columns
        for the min, mean, and SD mini-batch validation RMSEs.
    """

    # Define the initial parameters that were used to train the different
    # ensemble NN layer architectures
    params_dict = {
        'inputs':'predictions__', 'y':120, #trained with just indiv. model predictions on y_120
        'batch_size':128, 'lr':0.0001, 'weight_decay':1e-05}

    # Enter the directory of the NN ensemble model parameters and iterate through 
    # all layer architectures.
    param_dir = './parameters/ensemble'
    layers = [f for f in os.listdir(param_dir) if '.' not in f]

    # Generate the correct train and validation ensemble data.
    _, X_val, _, _, y_val, _, _, _, _, _ = data.get_ensemble_data(
        horizon=params_dict['y'])

    # Initialize the outputs.
    out = []
    metric_out = {'ensemble nn':
        {'min':None, 'mean':None, 'SD':None, 'n':None}}

    for layer in layers:

        # Get the inputs that define the NN model based on the current layer index.
        layer_idx = int(layer.split('_')[-1])
        layer_df_path = f'{param_dir}/ensemble_layers.pkl'
        layer_df = pd.read_pickle(layer_df_path)
        layer_inputs = layer_df.loc[layer_idx, 'ensemble_layers']
        
        # Construct the model based on this layer architecture.
        model = models.NN(
            input_dim=X_val.shape[1],
            **layer_inputs)
        
        # Read the parameters of each NN ensemble model within the 
        # folders of the correct inputs and horizons used in this training step.
        # Load these into the constructed model
        model_dir = f'{param_dir}/{layer}/{params_dict["inputs"]}/y_{params_dict["y"]}'
        model_file = f'batch_size({params_dict["batch_size"]})__lr({params_dict["lr"]})__' \
                    f'weight_decay({params_dict["weight_decay"]})__early_stopping(True).pt'
        model_path = f'{model_dir}/{model_file}'
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model = model.eval()

        # Calculate the full validation RMSE and add it to the output.
        y_hat = model(torch.from_numpy(X_val).float())
        y_hat = np.array(y_hat.squeeze(1).detach())
        errors = (y_hat - y_val) ** 2
        rmse = np.sqrt(np.mean(errors))
        out.append(rmse)

    # Calculate the min, mean, and SD of these aggregated full validation RMSEs.
    metric_out['ensemble nn']['min'] = min(out).round(2)
    metric_out['ensemble nn']['mean'] = np.mean(out).round(2)
    metric_out['ensemble nn']['SD'] = np.std(out).round(2)
    metric_out['ensemble nn']['n'] = len(out)

    # Aggregate the metrics into a pandas dataframe and return it
    df_ensemble_perf_layers = pd.DataFrame.from_dict(metric_out, orient='index')
    
    return df_ensemble_perf_layers


# Define a function that aggregates the min, mean, and SD of the full
# validation RMSEs of the ensemble neural networks across all ensemble inputs.
def get_ensemble_performance_of_inputs(horizon):
    """
    Function that calculates the full validation RMSE of every
    ensemble input combination for the neural network ensemble model. 
    It then outputs the min., mean, and SD of these RMSEs.

    Parameters
    ----------
    horizon : int
        Which horizon the models were trained on.

    Returns
    -------
    df_ensemble_perf_inputs : pd.DataFrame
        Pandas dataframe w/ one row and three columns
        for the min, mean, and SD mini-batch validation RMSEs.
    """

    # Define the hyperparameters that were used to train the NN ensemble models
    # across different inputs and horizons.
    batch_size = 128
    lr = '0.0001'
    weight_decay = 1e-05
    
    model_file = f'batch_size({batch_size})__lr({lr})__weight_decay({weight_decay})__' \
                'early_stopping(True).pt' #resulting filename of model weigths

    # Get the best layer parameters of the best performing NN ensemble model architecture.
    best_layers = get_best_ensemble_layers(
        batch_size=batch_size, #the same hyperparams were used to train diff
        lr=lr,                  #layer architectures as well as diff
        weight_decay=weight_decay) #inputs-horizons combinations
    
    # Also get the index that corresponds to this architecture.
    best_idx = get_best_ensemble_layers(
        batch_size=batch_size, 
        lr=lr,     
        weight_decay=weight_decay,
        return_idx=True)

    # Define the directory where each input combination is stored.
    input_dir = f'./parameters/ensemble/ensemble_layers_{best_idx}'

    # Initialize the outputs.
    out = []
    metric_out = {'ensemble nn':
        {'min':None, 'mean':None, 'SD':None, 'n':None}}

    # Iterate through each combination of ensemble inputs.
    inputs = [f for f in os.listdir(input_dir) if '.' not in f]
    for input in inputs:

        # Check which type of ensemble data should be generated.
        which_input = models.extract_ensemble_inputs(input)

        # Generate that particular ensemble data.
        _, X_val, _, _, y_val, _, _, _, _, _ = data.get_ensemble_data(
            horizon=horizon,
            **which_input) #e.g. include_time=True

        # Construct the model based on the shape of the input 
        # and using the best performing layer architecture.
        model = models.NN(
            input_dim=X_val.shape[1],
            **best_layers)

        # Load in the correct model parameters
        model_path = f'{input_dir}/{input}/y_{horizon}/{model_file}'
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model = model.eval()

        # Calculate the full validation RMSE and add it to our output.
        y_hat = model(torch.from_numpy(X_val).float())
        y_hat = np.array(y_hat.squeeze(1).detach())
        errors = (y_hat - y_val) ** 2
        rmse = np.sqrt(np.mean(errors))
        out.append(rmse)

    # Calculate the min, mean, and SD of these aggregated full validation RMSEs.
    metric_out['ensemble nn']['min'] = min(out).round(2)
    metric_out['ensemble nn']['mean'] = np.mean(out).round(2)
    metric_out['ensemble nn']['SD'] = np.std(out).round(2)
    metric_out['ensemble nn']['n'] = len(out)

    # Aggregate the metrics into a pandas dataframe and return it
    df_ensemble_perf_inputs = pd.DataFrame.from_dict(metric_out, orient='index')
    
    return df_ensemble_perf_inputs


# Define a function that aggregates the min, mean, and SD of the full
# validation RMSEs of the ensemble neural networks across all hyperparameters.
def get_ensemble_performance_of_hyperparams(horizon):
    """
    Function that calculates the full validation RMSE of every
    hyperparameter combination for the neural network ensemble model. 
    It then outputs the min., mean, and SD of these RMSEs.

    Parameters
    ----------
    horizon : int
        Which horizon the models were trained on.

    Returns
    -------
    df_ensemble_perf_hyperparams : pd.DataFrame
        Pandas dataframe w/ one row per datatype and three columns
        for the min, mean, and SD mini-batch validation RMSEs.
    """

    # Define the hyperparameters that were used to train the NN ensemble models
    # across different layer architectures, inputs, and horizons.
    batch_size = 128
    lr = 0.0001
    weight_decay = 1e-05

    # Get the index of the best performing layer architecture
    best_idx = get_best_ensemble_layers(
        batch_size=batch_size, 
        lr=lr,     
        weight_decay=weight_decay,
        return_idx=True)

    # Get the best model and best performing ensemble input combination
    # for the specified horizon.
    model, best_inputs = get_best_ensemble_model_by_horizon(
        horizon=horizon, 
        return_inputs=True,
        return_hyperparams=False,
        batch_size=batch_size, 
        lr=lr,     
        weight_decay=weight_decay)

    # Generate the correct validation data.
    _, X_val, _, _, y_val, _, _, _, _, _ = data.get_ensemble_data(
        horizon=horizon,
        **best_inputs)

    # Initialize the outputs.
    out = []
    metric_out = {'ensemble nn':
        {'min':None, 'mean':None, 'SD':None, 'n':None}}

    # Generate a string from the best_inputs dictionary that can be used
    # to enter the correct folder of the NN ensemble models.
    input_string = [k.replace('include_', '') for k, v in best_inputs.items()
                    if v is True] #e.g. filters out include_time=False
    input_string = '__'.join(input_string)
    input_string = 'predictions__' + input_string

    # Generate the directory containing all models of the best layer architecture
    # best input combination with the model weights of the hyperparameter combinations.
    hyper_dir = f'./parameters/ensemble/ensemble_layers_{best_idx}/{input_string}/y_{horizon}'
    hyper_files = [f for f in os.listdir(hyper_dir) if not f.startswith('.')]

    # Iterate through each hyperparam combo and read in the model parameters.
    for hyper_file in hyper_files:

        model_path = f'{hyper_dir}/{hyper_file}'
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model = model.eval()

        # Calculate the full validation RMSE and add it to our output.
        y_hat = model(torch.from_numpy(X_val).float())
        y_hat = np.array(y_hat.squeeze(1).detach())
        errors = (y_hat - y_val) ** 2
        rmse = np.sqrt(np.mean(errors))
        out.append(rmse)

    # Calculate the min, mean, and  of these aggregated full validation RMSEs.
    metric_out['ensemble nn']['min'] = min(out).round(2)
    metric_out['ensemble nn']['mean'] = np.mean(out).round(2)
    metric_out['ensemble nn']['SD'] = np.std(out).round(2)
    metric_out['ensemble nn']['n'] = len(out)

    # Aggregate the metrics into a pandas dataframe and return it.
    df_ensemble_perf_hyperparams = pd.DataFrame.from_dict(metric_out, orient='index')
    
    return df_ensemble_perf_hyperparams


# Define a function that aggregates the overall performance metrics, i.e. min.,
# mean, and SD (mini-batch) validation RMSEs in one dataframe for all training
# steps of the individual and ensemble models.
def get_overall_performance(horizon):
    """
    Function that uses the respective performance-aggregating function, such as
    get_indiv_performance_of_layers(), to aggregate the RMSE metrics across the
    different training steps for the individual and ensemble models.

    Parameters
    ----------
    horizon : int
        Which horizon the models were trained on. Only relevant for the training 
        step over input-horizon combos and hyperparameter tuning. That is, 
        because the training step that optimized the layer architecture,
        only use one fixed forecast horizon.
    
    Returns
    -------
    df_overall_perf : pd.DataFrame
        Pandas dataframe with one column for each model type and one row
        for each training step, e.g. hyperparameter tuning.
    """

    # Prepare the output to store all performance metrics.
    datatypes = ['images', 'images3d', 'irradiance', 'weather', 'combined']
    modeltypes = datatypes + ['ensemble nn']
    out = {modeltype:{
            'layer architecture':None, 'inputs & horizons':None, 'hyperparameters':None}
            for modeltype in modeltypes}

    # Get the metrics for the 'layer architecture' training steps for all datatypes.
    # Then, add those to the output.
    df = get_indiv_performance_of_layers()

    for datatype in datatypes:

        metrics = tuple(zip(df.loc[datatype, ['min', 'mean', 'SD']]))
        metrics = tuple([i[0] for i in metrics])
        out[datatype]['layer architecture'] = metrics

    # Repeat for the NN ensemble model.
    df = get_ensemble_performance_of_layers()
    metrics = tuple(zip(df.loc['ensemble nn', ['min', 'mean', 'SD']]))
    metrics = tuple([i[0] for i in metrics])
    out['ensemble nn']['layer architecture'] = metrics

    # Get the metrics of the 'inputs & horizons' training steps for all datatypes.
    # Then, add those to the output.
    df = get_indiv_performance_of_inputs(horizon=horizon)

    for datatype in datatypes:

        metrics = tuple(zip(df.loc[datatype, ['min', 'mean', 'SD']]))
        metrics = tuple([i[0] for i in metrics])
        out[datatype]['inputs & horizons'] = metrics

    # Repeat for the NN ensemble model.
    df = get_ensemble_performance_of_inputs(horizon=horizon)
    metrics = tuple(zip(df.loc['ensemble nn', ['min', 'mean', 'SD']]))
    metrics = tuple([i[0] for i in metrics])
    out['ensemble nn']['inputs & horizons'] = metrics

    # Get the metrics for the 'hyperparameters' training steps for all datatypes.
    # Then, add those to the output
    df = get_indiv_performance_of_hyperparams(horizon=horizon)

    for datatype in datatypes:

        metrics = tuple(zip(df.loc[datatype, ['min', 'mean', 'SD']]))
        metrics = tuple([i[0] for i in metrics])
        out[datatype]['hyperparameters'] = metrics 

    # Repeat for the NN ensemble model.
    df = get_ensemble_performance_of_hyperparams(horizon=horizon)
    metrics = tuple(zip(df.loc['ensemble nn', ['min', 'mean', 'SD']]))
    metrics = tuple([i[0] for i in metrics])
    out['ensemble nn']['hyperparameters'] = metrics

    # Create a dataframe from the output dictionary and return it.
    df_overall_perf = pd.DataFrame.from_dict(out)

    return df_overall_perf.transpose()


# Define a function that outputs the full train, val, and test
# RMSEs across all model types and forecast horizons.
def get_overall_train_val_test_errors(horizons):
    """
    Function that calculates the full train, val, and test
    RMSEs for all indiviual models, the linear ensemble baseline,
    and the NN ensemble models. It then outputs each of them
    in one large pandas dataframe.

    Parameters
    ----------
    horizons : int or 1d-array
        - If int, dataframe will contain only the train, val, test
        RMSEs for this horizon.
        - If 1d-array, dataframe will contain errors RMSEs for all horizons.

    Returns
    -------
    train_val_test_df : pd.DataFrame
        Pandas dataframe containing the train, val,
        and test RMSEs for all seven model types and all
        specified forecast horizons.
    """

    # Change the type of horizons to an array, if needed.
    if isinstance(horizons, int):
        horizons = [horizons]
    
    # Prepare a dictionary to store the output.
    datatypes = ['images', 'images3d', 'irradiance', 'weather', 'combined']
    modeltypes = datatypes + ['persistence', 'linear ensemble', 'NN ensemble']
    
    out = {modeltype:
            {f'y_{horizon}':{'Train':None, 
                            'Val':None,
                            'Test':None}
                for horizon in horizons}
            for modeltype in modeltypes}

    # Iterate through each horizon and generate the corresponding ensemble data.
    print('Calculating train, val, test RMSEs for the individual models for each '
        'specified forecast horizon...')

    for horizon in horizons:

        (X_train, X_val, X_test,
        y_train, y_val, y_test,
        t_train, t_val, t_test, _) = data.get_ensemble_data(
            horizon=horizon,
            return_as_df=True)

        # Iterate through each datatype and calculate the train, val, test RMSEs.
        for datatype in datatypes:

            # Calculate the squared forecast errors.
            # For the training dataset.
            yhat_train = X_train[datatype].to_numpy()
            train_errors = (yhat_train - y_train)**2

            # For the validation set.
            yhat_val = X_val[datatype].to_numpy()
            val_errors = (yhat_val - y_val)**2

            # For the test set.
            yhat_test = X_test[datatype].to_numpy()
            test_errors = (yhat_test - y_test)**2

            # Calculate the RMSEs for all three datasets.
            rmse_train = np.sqrt(np.mean(train_errors))
            rmse_val = np.sqrt(np.mean(val_errors))
            rmse_test = np.sqrt(np.mean(test_errors))

            # Add them to the output.
            out[datatype][f'y_{horizon}']['Train'] = rmse_train
            out[datatype][f'y_{horizon}']['Val'] = rmse_val
            out[datatype][f'y_{horizon}']['Test'] = rmse_test

    # We now also want the best errors of the linear ensemble baseline.
    # For this, calculate the errors for all possible ensemble input combos.
    rmses_linear_df = df = compare_linear_combinations(
        horizons=horizons,
        include_test_errors=True) #we want the train, val, and test RMSEs

    # Iterate through all horizons and find which ensemble inputs yields the best
    # validation RMSE of each forecast horizon.
    for horizon in horizons:

        best_input = rmses_linear_df.loc[:, (f'y_{horizon}', 'Val RMSE')].idxmin()

        # Use this best-performing input to the linear combination baseline to extract
        # its corresponding train, val, and test RMSEs
        rmse_linear_train = rmses_linear_df.loc[best_input, (f'y_{horizon}', 'Train RMSE')]
        rmse_linear_val = rmses_linear_df.loc[best_input, (f'y_{horizon}', 'Val RMSE')]
        rmse_linear_test = rmses_linear_df.loc[best_input, (f'y_{horizon}', 'Test RMSE')]

        # Add these errors to the output.
        out['linear ensemble'][f'y_{horizon}']['Train'] = rmse_linear_train
        out['linear ensemble'][f'y_{horizon}']['Val'] = rmse_linear_val
        out['linear ensemble'][f'y_{horizon}']['Test'] = rmse_linear_test

    # Lastly, we need the best RMSEs for the NN ensemble models.
    # For this, we need to make some preparations. We need
    # the best performing model, input, and hyperparameter combos:

    # Define which horizons belong to which horizon bucket.
    intra_hours = [i for i in horizons if i < 60] #everything < 1 hour
    intra_days = [i for i in horizons if i >=60 and i < 24*60] #everything < 1 day
    day_aheads = [i for i in horizons if i >= 24*60] #everything > 1 day

    # Get the best performing inputs and hyperparams for the intra-hour horizons.
    # But only if we need it for the specified horizons.
    if len(intra_hours) > 0:
        model_y30, ensemble_inputs_y30, ensemble_hyper_y30 = get_best_ensemble_model_by_horizon(
            horizon=30, return_inputs=True, return_hyperparams=True, #return hyperparams
            batch_size=None, lr=None, weight_decay=None) #search across all hyperparams
    else:
        model_y30 = ensemble_inputs_y30 = ensemble_hyper_y30 = None

    # Get the best performing inputs and hyperparams for the intra-day horizons.
    # But only if we need it for the specified horizons.
    if len(intra_days) > 0:
        model_y120, ensemble_inputs_y120, ensemble_hyper_y120 = get_best_ensemble_model_by_horizon(
            horizon=120, return_inputs=True, return_hyperparams=True, #return hyperparams
            batch_size=None, lr=None, weight_decay=None) #search across all hyperparams
    else:
        model_y120 = ensemble_inputs_y120 = ensemble_hyper_y120 = None

    # Get the best performing inputs and hyperparams for the day-ahead horizons.
    # But only if we need it for the specified horizons.
    if len(day_aheads) > 0:
        model_y1440, ensemble_inputs_y1440, ensemble_hyper_y1440 = get_best_ensemble_model_by_horizon(
            horizon=1440, return_inputs=True, return_hyperparams=True, #return hyperparams
            batch_size=None, lr=None, weight_decay=None) #search across all hyperparams
    else:
        model_y1440 = ensemble_inputs_y1440 = ensemble_hyper_y1440 = None

    # Also get the index of the best performing layer architecture.
    best_idx = get_best_ensemble_layers(
        batch_size=128, #we used these hyperparams to find the best architecture 
        lr=0.0001,     
        weight_decay=1e-05,
        return_idx=True)

    # Define a dictionary that denotes which horizons need which models.
    horizon_dict = {
        'intra_hour':{'model':model_y30, 'inputs':ensemble_inputs_y30, 'hyper':ensemble_hyper_y30},
        'intra-day':{'model':model_y120, 'inputs':ensemble_inputs_y120, 'hyper':ensemble_hyper_y120},
        'day-ahead':{'model':model_y1440, 'inputs':ensemble_inputs_y1440, 'hyper':ensemble_hyper_y1440}}

    # Iterate through each horizon and use the correct model, inputs, and hyperparameters.
    for horizon in horizons:

        if horizon in intra_hours:
            model = horizon_dict['intra_hour']['model']
            inputs = horizon_dict['intra_hour']['inputs']
            hyperparams = horizon_dict['intra_hour']['hyper']
        
        elif horizon in intra_days:
            model = horizon_dict['intra-day']['model']
            inputs = horizon_dict['intra-day']['inputs']
            hyperparams = horizon_dict['intra-day']['hyper']

        elif horizon in day_aheads:
            model = horizon_dict['day-ahead']['model']
            inputs = horizon_dict['day-ahead']['inputs']
            hyperparams = horizon_dict['day-ahead']['hyper']

        # Generate a string from the best_inputs dictionary that can be used
        # to enter the correct folder of the NN ensemble models.
        input_string = [k.replace('include_', '') for k, v in inputs.items()
                        if v is True] #e.g. filters out include_time=False
        input_string = '__'.join(input_string)
        input_string = 'predictions__' + input_string

        # Generate the correct ensemble validation data.
        (X_train, X_val, X_test,
        y_train, y_val, y_test,
        _, _, _, _) = data.get_ensemble_data(
            horizon=horizon,
            **inputs)

        # Generate the directory containing all models of the best layer architecture
        # best input combination with the model weights of the hyperparameter combinations.
        hyper_dir = f'./parameters/ensemble/ensemble_layers_{best_idx}/{input_string}/y_{horizon}'
        hyper_file = f'batch_size({hyperparams["batch_size"]})__lr({hyperparams["lr"]})__' \
                    f'weight_decay({hyperparams["weight_decay"]})__early_stopping(True).pt'
        
        # Load in the weights into the model
        model_path = f'{hyper_dir}/{hyper_file}'
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model = model.eval()

        # Calculate the full training RMSE.
        yhat_train = model(torch.from_numpy(X_train).float())
        yhat_train = np.array(yhat_train.squeeze(1).detach())
        train_errors = (yhat_train - y_train) ** 2
        rmse_train = np.sqrt(np.mean(train_errors))

        # Repeat for the validation RMSE.
        yhat_val = model(torch.from_numpy(X_val).float())
        yhat_val = np.array(yhat_val.squeeze(1).detach())
        val_errors = (yhat_val - y_val) ** 2
        rmse_val = np.sqrt(np.mean(val_errors))

        # And for the test RMSEs.
        yhat_test = model(torch.from_numpy(X_test).float())
        yhat_test = np.array(yhat_test.squeeze(1).detach())
        test_errors = (yhat_test - y_test) ** 2
        rmse_test = np.sqrt(np.mean(test_errors))

        # Add them all to the output.
        out['NN ensemble'][f'y_{horizon}']['Train'] = rmse_train
        out['NN ensemble'][f'y_{horizon}']['Val'] = rmse_val
        out['NN ensemble'][f'y_{horizon}']['Test'] = rmse_test

    # Finally, we want the RMSEs of the persistence baselines.
    # For this, we first need the correct data for each horizon 
    # so we can feed it into the model.
    print('Extracting, preprocessing, and predicting on the irradiance data '
        'using the persistence model...')
    dataset = data.create_dataset(type='irradiance', from_pkl=True)
    model = models.Persistence()
    
    for horizon in horizons:
        print(f'...with forecast horizon {horizon}...')
        X, y, t = dataset.get_data(
            datatype='irradiance',
            forecast_horizon=horizon,
            sample=False,
            sample_size=1,
            delta=1,
            length=10)

        # Split the data into train, val, and test sets:
        (X_train, X_val, X_test,
        y_train, y_val, y_test,
        _, _, _) = preprocessing.split_and_preprocess(
            X=X, y=y, timestamps=t, preprocessing=None)

        # Calculate the train RMSE.
        yhat_train = model(X_train)
        yhat_train = yhat_train.detach().numpy().flatten()
        train_errors = (y_train.numpy() - yhat_train) ** 2
        rmse_train = np.sqrt(np.mean(train_errors))

        # Repeat for the val RMSE.
        yhat_val = model(X_val)
        yhat_val = yhat_val.detach().numpy().flatten()
        val_errors = (y_val.numpy() - yhat_val) ** 2
        rmse_val = np.sqrt(np.mean(val_errors))

        # And for the test RMSE.
        yhat_test = model(X_test)
        yhat_test = yhat_test.detach().numpy().flatten()
        test_errors = (y_test.numpy() - yhat_test) ** 2
        rmse_test = np.sqrt(np.mean(test_errors))

        # Add them all to the output.
        out['persistence'][f'y_{horizon}']['Train'] = rmse_train
        out['persistence'][f'y_{horizon}']['Val'] = rmse_val
        out['persistence'][f'y_{horizon}']['Test'] = rmse_test

    # Prepare an output that is shaped for a multi-level column in pandas.
    output = {modeltype:{} for modeltype in out.keys()}

    # Iterate through each model type and fill that output.
    for modeltype, v in out.items():

        # Extract each horizon and subcolumn.
        for horizon, subcol in v.items():

            # Create a tuple of the horizon and subcol name and 
            # assign it the corresponding value.
            for subcol_name, val in subcol.items():
                output[modeltype][(horizon, subcol_name)] = val

    # Create a multi-level column df containing the RMSEs per horizon and model type
    # and return it.
    train_val_test_df = pd.DataFrame.from_dict(output, orient='index')

    return train_val_test_df.round(2)


# Define a function that extracts the hyperparameters from a filename where 
# the train and val losses of a given model are stored.
def extract_hyperparameters(hyperparam_name, as_string=True):
    """
    Function that extracts the hyperparameters from a 
    hyperparam_name that we use in filenames containing 
    the train and validation losses.

    Parameters
    ----------
    hyperparam_name : str
        String in the format used to store train and val losses:
        'batch_size(a)__lr(b)__weight_decay(c)'
    as_string : bool
        - If True, the values of the hyperparameters will be strings.
        - If False, the values of the hyperparameters will be numerical.

    Returns
    -------
    hyperparam_dict : dict
        Dictionary containing the hyperparameters and their
        values.
    """

    # Split the hyperparameters and isolate the number.
    hyperparams = hyperparam_name.split('__')
    hyperparams = [h.replace('(', ' ') for h in hyperparams]
    hyperparams = [h.replace(')', '') for h in hyperparams]

    # Extract the number and save everything into a dict.
    hyperparam_dict = {}
    for hyperparam in hyperparams:
        hyperparam, num = hyperparam.split(' ')

        # Convert the string to numerical if specified
        if as_string == False:
            try:
                num = int(num)
            except ValueError:
                num = float(num)
            
        hyperparam_dict[hyperparam] = num
    
    return hyperparam_dict


# Define a function to extract a value of a layer parameter
# from a dictionary (used in a pandas.DataFrame).
def extract_layer_param(x, param):
    
    # If the parameter exist, return the value of it
    try:
        return x[param]
    
    # Otherwise, return None to indicate the non-existence.
    except KeyError:
        return None



    
            
