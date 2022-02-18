"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plotting.py defines the functions that are used
to create graphs to understand the performances 
of a variety of trained models.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# Import general python modules
import math
import gc
import itertools
import copy
import os
import random
import pytz
from scipy import stats

# Import plotting, vector, and ML modules
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.dates import DateFormatter
from matplotlib import cm
from matplotlib.patches import Rectangle
from pandas.tseries.offsets import WeekOfMonth
import seaborn as sns
import pandas as pd
import numpy as np
import torch

# Import our own modules that are needed
import data
import preprocessing
import getperformance
import models
#importlib.reload(getperformance)


# Define a function that plots the best train and validation 
# mini-batch RMSEs for each datatype in one figure.
def plot_best_train_val(datatypes, deltas, lengths, horizon,
                        batch_sizes, lrs, weight_decays):
    """
    Function that first finds the best layer architecture for the given
    delta, length, etc. per datatype. It then plots the training
    and validation loss per epoch for each of these models.

    Parameters
    ----------
    datatypes : str or 1d-array
                {'images', 'images3d', 'irradiance', 'weather', 'combined'}
        - If str, plots train and val losses of this particular datatype.
        - If 1d-array, plots train and val losses for each
        of the specified datatypes.
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

    Output
    ------
    None
        Only plots a graph.
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

    # Prepare figure title.
    if horizon < 60:
        forecast = f'{horizon}-Minutes-Ahead Forecast'
    elif horizon == 60:
        forecast = f'1-Hour-Ahead Forecast'
    elif horizon < 60*24:
        forecast = f'{int(horizon/60)}-Hours-Ahead Forecast'
    elif horizon == 60*24:
        forecast = f'1-Day-Ahead Forecast'
    else:
        forecast = f'{int(horizon/60/24)}-Days-Ahead Forecast'
    
    # Initilaize figure and axis.
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title(
        f'Best {forecast} Train and Validation RMSE Per Datatype',
        fontsize=20)
    colors = cm.Set1.colors
    ax.set_xlabel('Epochs', fontsize=16)
    ax.set_ylabel('Mean Mini-Batch RMSE (W/m^2)', fontsize=16)
    ax.tick_params(labelsize=14)
    
    # For each datatype and the given delta, length, etc., 
    # we need to now find the best performing layer architecture.
    for i, datatype in enumerate(datatypes):

        best_layer = getperformance.get_best_layers(
            datatype=datatype, return_layer_params=False,
            delta=deltas[i], length=lengths[i],
            horizon=horizon, batch_size=batch_sizes[i], 
            lr=lrs[i], weight_decay=weight_decays[i],
            output_missing=False)

        #Retrieve the train and val losses for this layer index
        # and the given delta, length, etc.
        loss_path = f'./performance/{datatype}/{datatype}_layers_{best_layer}/' \
                    f'delta_{deltas[i]}/length_{lengths[i]}/y_{horizon}/' \
                    f'batch_size({batch_sizes[i]})__lr({lrs[i]})__' \
                    f'weight_decay({weight_decays[i]}).pkl'
        
        loss_df = pd.read_pickle(loss_path)
        n_epochs = loss_df.index.max()

        # Plot the train and validation loss per epoch.
        ax.plot(
            range(1, n_epochs+2), 
            loss_df['train'],
            marker='_',
            markersize=20,
            label=f'Train RMSE ({datatypes[i]})',
            color=tuple(0.65*x for x in colors[i]))
        ax.plot(
            range(1, n_epochs+2), 
            loss_df['val'],
            marker='o',
            label=f'Validation RMSE ({datatypes[i]})',
            color=colors[i])

        print(datatype, 'Min Train Error: ', loss_df['train'].to_numpy().min().round(2))
        print(datatype, 'Min Val Error: ', loss_df['val'].to_numpy().min().round(2))

    # Final enhancement to figure.
    plt.legend(loc='upper right', fontsize=16)
    fig.tight_layout()


# Define a function that plots a histogram of the validation mini-batch RMSEs
# of a specified model, input data, forecast horizon, and hyperparameters.
def plot_rmse_hist(datatypes, deltas, lengths, horizon,
                    batch_sizes, lrs, weight_decays, min_or_mean):
    """
    Function that aggregates all validation RMSEs of a given 
    model layer architecture for each given datatype, delta, length,
    forecast horizon, and hyperparameters.
    It then plots a histogram of these RMSEs for each of these
    datatypes.

    Parameters
    ----------
    datatypes : str or 1d-array
                {'images', 'images3d', 'irradiance', 'weather', 'combined'}
        - If str, plots RMSE histogram of this particular datatype.
        - If 1d-array, plots one RMSE histogram for each
        of the specified datatypes.
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
    min_or_mean : {'min', 'mean'}
        - If min, aggregates the RMSEs using the best RMSEs per model.
        - If mean, aggregates the RMSEs using the mean of the lowest 5 RMSES
        per model.

    Output
    ------
    None
        Only plots a graph.
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

    # Prepare shape of the subplot grid. We'll use max. 3 columns
    # and calculate the number of rows from that.
    max_cols = 3

    if len(datatypes) >= max_cols:
        ncols = max_cols
    else:
        ncols = len(datatypes) 
    nrows = math.ceil(len(datatypes) / max_cols)

    # Prepare figure title.
    if horizon < 60:
        forecast = f'{horizon}-Minutes-Ahead Forecast'
    elif horizon == 60:
        forecast = f'1-Hour-Ahead Forecast'
    elif horizon < 60*24:
        forecast = f'{horizon/60}-Hours-Ahead Forecast'
    elif horizon == 60*24:
        forecast = f'1-Day-Ahead Forecast'
    else:
        forecast = f'{horizon/60/24}-Days-Ahead Forecast'
    
    # Prepare the figure.
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(ncols*7.5, nrows*6),
                            sharex=True)
    fig.suptitle(f'{forecast} Validation RMSEs Per Datatype',
                fontsize=20)
    try:
        axs = axs.flatten()
    except AttributeError:
        axs = [axs] #needed if only one element in datatypes was given

    # Iterate through each datatype and get the dataframe containing 
    # validation set performance of the specified trained models.
    # Then plot a histogram for this particular datatype.
    for i, datatype in enumerate(datatypes):
        df = getperformance.get_layers_performance(
            datatype=datatype, 
            delta=deltas[i], 
            length=lengths[i],
            horizon=horizon, 
            batch_size=batch_sizes[i], 
            lr=lrs[i], 
            weight_decay=weight_decays[i],
            output_missing=False, 
            min_or_mean=min_or_mean)
        df = df[df['Best Validation RMSE'].notnull()]
        
        # Plot the histogram.
        iqr = stats.iqr(df['Best Validation RMSE'].to_numpy()) #interquartile range
        bin_width = 2 * iqr * (len(df)**(-1/3)) #Freedman–Diaconis rule
        num_bins = (df['Best Validation RMSE'].max()-df['Best Validation RMSE'].min()) / bin_width
        axs[i].hist(x=df['Best Validation RMSE'], bins=int(num_bins))
        axs[i].set_title(datatype, fontsize=16)
        axs[i].set_xlabel('Mean Mini-Batch Validation RMSE (W/m^2)', fontsize=16)

        if i % max_cols == 0:
            axs[i].set_ylabel('Frequency', fontsize=16)
        axs[i].tick_params(labelsize=14)

    # Hide the remaining empty subplots.
    num_remaining = len(axs) - len(datatypes)
    for i in range(len(axs)-num_remaining, len(axs)):
        axs[i].axis('off')
    
    fig.tight_layout()


# Define a function that plots the true target y
# vs the predicted yhat.
def plot_ytrue_yhat(datatypes, deltas, lengths, horizon,
                    batch_sizes, lrs, weight_decays):
    """
    Function that first finds the best layer architecture for the given
    delta, length, etc. per datatype. It then plots a scatterplot of the
    true target y vs. the predicted y_hat for each model.

    Parameters
    ----------
    datatypes : str or 1d-array
                {'images', 'images3d', 'irradiance', 'weather', 'combined'}
        - If str, scatterplot of this particular datatype.
        - If 1d-array, plots one scatterplot for each
        of the specified datatypes.
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

    Output
    ------
    None
        Only plots a graph.
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

    # We will use a sample size of 0.5% as otherwise
    # the scatterplot becomes unreadable.
    sample_size = 0.005

    # Prepare shape of the subplot grid. We'll use max. 3 columns
    # and calculate the number of rows from that.
    max_cols = 3

    if len(datatypes) >= max_cols:
        ncols = max_cols
    else:
        ncols = len(datatypes) 
    nrows = math.ceil(len(datatypes) / max_cols)

    # Prepare figure title.
    if horizon < 60:
        forecast = f'{horizon}-Minutes-Ahead Forecast'
    elif horizon == 60:
        forecast = f'1-Hour-Ahead Forecast'
    elif horizon < 60*24:
        forecast = f'{horizon/60}-Hours-Ahead Forecast'
    elif horizon == 60*24:
        forecast = f'1-Day-Ahead Forecast'
    else:
        forecast = f'{horizon/60/24}-Days-Ahead Forecast'
    
    # Prepare figure and axes.
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(ncols*7.5, nrows*6),
                            sharey=True)
    fig.suptitle(f'{forecast} vs. True Target Per Datatype',
                fontsize=20)
    try:
        axs = axs.flatten()
    except AttributeError:
        axs = [axs] #needed if only one element in datatypes was given

    # For each datatype and the given delta, length, etc., 
    # we need to now find the best performing layer architecture.   
    for i, datatype in enumerate(datatypes):

        best_layer = getperformance.get_best_layers(
            datatype=datatype, return_layer_params=False,
            delta=deltas[i], length=lengths[i],
            horizon=horizon, batch_size=batch_sizes[i], 
            lr=lrs[i], weight_decay=weight_decays[i])

        # Then, load in the respective model.
        model = models.load_trained_model(
            datatype=datatype,
            layers_idx=best_layer,
            delta=deltas[i],
            length=lengths[i],
            horizon=horizon,
            batch_size=batch_sizes[i],
            lr=lrs[i],
            weight_decay=weight_decays[i])

        # Get the correct data depending on the datatype.
        # Also apply the correct preprocessing.
        print(f'Extracting and preprocessing the {datatype} data.')
        dataset = data.create_dataset(
            type=datatype,
            from_pkl=True)
        dataset.create_sample(size=sample_size, seed=42)

        X, y, t = dataset.get_data(
            datatype=datatype,
            forecast_horizon=horizon,
            sample=True,
            sample_size=sample_size,
            delta=deltas[i],
            length=lengths[i])
                             
        (X_train, X_val, X_test,
        y_train, y_val, y_test,
        t_train, t_val, t_test) = preprocessing.split_and_preprocess(
            X=X, y=y, timestamps=t, preprocessing=datatype)
      
        # Clean memory as we only need the validation sets.
        del dataset, X, y, t, X_train, y_train, t_train
        del X_test, y_test, t_test
        gc.collect()

        # Get predictions of the model and calculate the squared errors.
        yhat = model(X_val)
        yhat = yhat.detach().numpy().flatten()
        y_val = y_val.detach().numpy().flatten()

        # Make all dots red where yhat has a negative values
        # as a prediction of negative Watt/m^2 clearly
        # does not make sense.
        red = cm.Set1.colors[0]
        blue = cm.Set1.colors[1]
        colors = [red if y<0 else blue for y in yhat]
        
        # Plot the true vs predicted y.
        axs[i].scatter(x=y_val, y=yhat, c=colors)
        axs[i].set_xlabel('True y (W/m^2)', fontsize=16)
        axs[i].tick_params(labelsize=14)
        axs[i].set_title(datatype, fontsize=18)
        if i % max_cols == 0:
            axs[i].set_ylabel('Predicted y (W/m^2)', fontsize=16)

        # Also print the correlation between ytrue and yhat.
        corr = stats.pearsonr(y_val, yhat)
        print(f'Pearson correlation of {datatype} model: {corr[0].round(2)}')
        print('----------------------------------')

    # Hide the remaining empty subplots.
    num_remaining = len(axs) - len(datatypes)
    for i in range(len(axs)-num_remaining, len(axs)):
        axs[i].axis('off')
    
    fig.tight_layout()


# Define a function to compare the performance for different values
# of a part of a layer architecture of a model.
def plot_performance_by_layer_parts(layer_parts, datatype, delta, length, horizon,
                                    batch_size, lr, weight_decay):
    """
    Function that uses the dataframe of getperformance.get_layers_performance()
    to compare the validation set mini-batch performance across different values
    of specific parameter(s) of the network layer, e.g. the kernel size in a CNN.

    Parameters
    ----------
    layer_parts : str or 1d-array
                {'cnn_num', 'conv_filters', 'conv_strides', 
                'lstm_hidden_size', 'lstm_num_layers, 'lstm_dropout',
                'dense_num', 'dense_dropout'}
        - If str, plots one graph with the best validation RMSE 
        for the values of the specified layer part.
        - If 1d-array, plots multiple graphs with the best validation 
        mini-batch RMSE for the values of the specified layer parts.
    datatype : {'images', 'images3d', 'irradiance', 'weather', 'combined'}
        Which datatype was used to train the models whose
        performances are to be checked.
    delta : int
        Which delta (in minutes) was between the datapoints
        of the X data sequence used to train the models.
    length : int
        How long the sequence of the X data used to train the model.
    horizon : int
        The forecast horizon of the y target the models were 
        trained on.
    batch_size : int
        What batch sizes were used to train the models.
    lr : float
        What learning rates were used to train the models.
    weight_decay : float
        What weight decay was used to train the models.

    Returns
    -------
    None
        Only plots a graph/multiple graphs of the validation set performance across
        the values of the parameters of the specified layer part(s).
    """

    # Check if a single or multiple layer parts were given.
    if isinstance(layer_parts, str):
        layer_parts = [layer_parts]
    
    # Prepare shape of the subplot grid. We'll use max. 3 columns
    # and calculate the number of rows from that.
    max_cols = 3

    if len(layer_parts) >= max_cols:
        ncols = max_cols
    else:
        ncols = len(layer_parts) 
    nrows = math.ceil(len(layer_parts) / max_cols)

    # Prepare figure and axes.
    if 'images' in datatype:
        modeltype = 'CNN'
    else:
        modeltype = 'LSTM' 
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(ncols*7.5, nrows*6),
                            sharey=True)
    fig.suptitle(f'Best Validation RMSEs for different parameters \n'
                f'of the {modeltype} using {datatype} data with \n'
                f'delta: {delta}, length: {length}, y: {horizon}',
                fontsize=20)
    try:
        axs = axs.flatten()
    except AttributeError:
        axs = [axs] #needed if only one element in layer_parts was given

    # Get the dataframe containing validation set performance
    # of the specified trained models.
    df = getperformance.get_layers_performance(
        datatype=datatype, 
        delta=delta, 
        length=length,
        horizon=horizon, 
        batch_size=batch_size, 
        lr=lr, 
        weight_decay=weight_decay,
        output_missing=False,
        min_or_mean='mean') #we used no early stopping when training
                            #the different layer architectures
                            #so we should aggregate the RMSEs by using
                            #the mean of the lowest RMSEs per model.

    # Iterate through all specified layer parts and add a column 
    # to the df containing the values of the specific layer part.
    for i, layer_part in enumerate(layer_parts):

        if isinstance(df.loc[0,], list):
            col_name = f'Average {layer_part}'
        else:
            col_name = layer_part
        df[col_name] = df['layers'].apply(lambda x: np.mean(x[layer_part]))
                
        # Get a list of RMSEs for each value of this particular layer part.
        layer_part_vals = sorted(df[col_name].unique())
        layer_part_rmses = []
        for j in layer_part_vals:
            rmses = df[(df[col_name] == j) & (df['Best Validation RMSE'].notnull())]
            rmses = rmses['Best Validation RMSE'].to_numpy()
            layer_part_rmses.append(rmses)

        # Modify the xtick labels to make them more readable.
        if layer_part == 'conv_filters':
            #x_ticklabels = [int(i) if i % 4 == 0 else None for i in layer_part_vals ]
            x_ticklabels = [int(v) if i%3==2 else None for i, v in enumerate(layer_part_vals)]
        else:
            x_ticklabels = [int(i) if i.is_integer() else i for i in layer_part_vals]
        
        # Plot a boxplot for each value of this particular layer part.
        axs[i].boxplot(layer_part_rmses)
        axs[i].set_xlabel(col_name, fontsize=14)
        axs[i].set_xticklabels(x_ticklabels)
        axs[i].tick_params(labelsize=14)
        if i % max_cols == 0:
            axs[i].set_ylabel('Mean Mini-Batch RMSE (W/m^2)', fontsize=14)

    # Hide the remaining empty subplots.
    num_remaining = len(axs) - len(layer_parts)
    for i in range(len(axs)-num_remaining, len(axs)):
        axs[i].axis('off')

    fig.tight_layout()


# Define a function that plots the mini-batch val. RMSE for different forecast
# horizons by delta and length for each datatype used to 
# train the LSTM models.
def plot_lstm_error_by_delta_length(datatypes, horizons, batch_sizes,
                                    lrs, weight_decays, min_or_mean='min'):
    """
    Function that gets the best mini-batch validation RMSE per horizon 
    for each delta and length and then plots it for each datatype 
    used to train the LSTM models.

    Parameters
    ----------
     datatypes : str or 1d-array
                {'irradiance', 'weather', 'combined'}
        - If str, plots RMSE by delta and length for this datatype.
        - If 1d-array, plots RMSE by delta and length for each
        of the specified datatypes.
    horizons : int or 1d-array
        - If int, the single specified forecast horizon will be plotted
        by delta and length.
       - If 1d-array, all specified horizons will be plotted
       by delta and length.
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
    min_or_mean : {'min', 'mean'} (Default: 'min')
        - If min, plots the minimum RMSE per delta/length and horizon.
        - If mean, plots the mean RMSE per delta/length and horizon.
    """

    # Change the type of some of the parameters to an array, if needed.
    if isinstance(datatypes, str):
        datatypes = [datatypes]
    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes]
    if isinstance(lrs, float):
        lrs = [lrs]
    if isinstance(weight_decays, float):
        weight_decays = [weight_decays]

    # Prepare figure and axes.
    fig, axs = plt.subplots(
        nrows=2, 
        ncols=len(datatypes), 
        figsize=(6*len(datatypes), 8),
        sharey=True)
    if min_or_mean == 'min':
        fig.suptitle(f'Min. RMSE by Horizon, Delta, Length, and Datatype',
                    fontsize=20)
    elif min_or_mean == 'mean':
        fig.suptitle(f'Mean RMSE by Horizon, Delta, Length, and Datatype',
                    fontsize=20)
    colors = cm.Set2.colors
    markers = ['o', 'v', 's']

    # Set the list of all deltas and lengths that shall be searched.
    all_deltas = [1, 3, 5, 10, 15]
    all_lengths = [10, 25, 50, 100]

    # For each datatype, find the performance for all horizons,
    # deltas and lengths.
    for i, datatype in enumerate(datatypes):

        for j, horizon in enumerate(horizons):

            df = None 

            for delta, length in itertools.product(all_deltas, all_lengths):

                # Get the best RMSEs and aggregate the data.
                perf = getperformance.get_layers_performance(
                    datatype=datatype,
                    delta=delta,
                    length=length,
                    horizon=horizon,
                    batch_size=batch_sizes[i],
                    lr=lrs[i],
                    weight_decay=weight_decays[i],
                    output_missing=False,
                    min_or_mean='min') #we used early stopping when training
                                        #all delta-length-horizon combinations
                                        #so we can just use the best RMSE here
                
                # Either replace the dataframe
                # or combine both of them.
                if df is None:
                    df = perf
                else:
                    df = pd.concat([df, perf])

            # Clean and aggregate the performance data
            # by delta length.
            df = df[df['Best Validation RMSE'].notnull()]
            
            # Either plot the mean RMSE across deltas/lengths or just
            # the best/minimum RMSE.
            if min_or_mean == 'min':
                df_delta = df.groupby(by=['delta']).std()[['Best Validation RMSE']]
                df_delta.reset_index(inplace=True)
                df_length = df.groupby(by=['length']).std()[['Best Validation RMSE']]
                df_length.reset_index(inplace=True)
            elif min_or_mean == 'mean':
                df_delta = df.groupby(by=['delta']).mean()[['Best Validation RMSE']]
                df_delta.reset_index(inplace=True)
                df_length = df.groupby(by=['length']).mean()[['Best Validation RMSE']]
                df_length.reset_index(inplace=True)
            
            # Plot the RMSE by delta on the top subplot(s).
            axs[0, i].plot(
                df_delta['delta'],
                df_delta['Best Validation RMSE'],
                color=colors[j],
                marker=markers[j],
                label=f'horizon: {horizon}')
            axs[0, i].tick_params(labelsize=14)
            axs[0, i].set_title(datatype, fontsize=16)
            axs[0, i].set_xlabel(
                'delta (Minutes)', fontsize=14)
            if i == 0:
                axs[0, i].set_ylabel(
                    'Mean Mini-Batch RMSE (W/m^2)', fontsize=14)

            # Plot the RMSE by length on the top subplot(s).
            axs[1, i].plot(
                df_length['length'],
                df_length['Best Validation RMSE'],
                color=colors[j],
                marker=markers[j],
                label=f'horizon: {horizon}')
            axs[1, i].tick_params(labelsize=14)
            axs[1, i].set_title(datatype, fontsize=16)
            axs[1, i].set_xlabel(
                'length', fontsize=14)
            if i == 0:
                if min_or_mean == 'min':
                    axs[1, i].set_ylabel(
                        'Min. Mini-Batch RMSE (W/m^2)', fontsize=14)
                elif min_or_mean == 'mean':
                    axs[1, i].set_ylabel(
                        'Mean Mini-Batch RMSE (W/m^2)', fontsize=14)

    # Final figure adjustments.
    plt.legend(loc='best', fontsize=16)
    fig.tight_layout()


# Define a function that plots the mini-batch val. RMSE for different forecast
# horizons by delta for the 3D CNN model.
def plot_cnn_error_by_delta(horizons, batch_size, lr, weight_decay):
    """
    Function that plots the mini-batch validation RMSE per horizon 
    of the 3D CNN model for each delta.

    Parameters
    ----------
    horizons : int or 1d-array
        - If int, the single specified forecast horizon will be plotted
        by delta and length.
       - If 1d-array, all specified horizons will be plotted
       by delta and length.
    batch_size : int
        What batch size was used to train the model
    lr : float
        What learning rate was used to train the model.
    weight_decay : float
        What weight decay was used to train the model.

    Returns
    -------
    None
        Only plots a graph.

    """

    # Change the type of horizons if needed.
    if isinstance(horizons, int):
        horizons = [horizons]
        
    # Set the list of all deltas of the image sequences 
    # that shall be searched.
    deltas3d = [10, 30, 60]

    # Prepare figure and axis.
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(f'RMSE by Horizon and Delta of the 3D CNN',
                fontsize=20)
    colors = cm.Set2.colors
    markers = ['o', 'v', 's']

    # Iterate through each horizon and aggregate the RMSEs.
    for i, horizon in enumerate(horizons):

        df = None 

        # Aggregate the RMSEs for each delta.
        for delta in deltas3d:

            # Get the best RMSEs and aggregate the data.
            perf = getperformance.get_layers_performance(
                datatype='images3d',
                delta=delta,
                length=3, #we only looked at image trios
                horizon=horizon,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                output_missing=False,
                min_or_mean='min') #we used early stopping when training
                                    #all delta-horizon combinations
                                    #so we can just use the best RMSE here
            
            # Either replace the dataframe
            # or combine both of them.
            if df is None:
                df = perf
            else:
                df = pd.concat([df, perf])

        # Clean and aggregate the performance data
        # by delta length.
        df = df[df['Best Validation RMSE'].notnull()]
        df_delta = df.groupby(by=['delta']).mean()[['Best Validation RMSE']]
        df_delta.reset_index(inplace=True)

        # Plot the RMSE by delta.
        ax.plot(
            df_delta['delta'],
            df_delta['Best Validation RMSE'],
            color=colors[i],
            marker=markers[i],
            label=f'horizon: {horizon}')
        ax.tick_params(labelsize=14)
        ax.set_title('images', fontsize=16)
        ax.set_xlabel(
            'delta (Minutes)', fontsize=14)
        ax.set_ylabel(
            'Mean Mini-Batch RMSE (W/m^2)', fontsize=14)

    # Final figure adjustments.
    plt.legend(loc='best', fontsize=16)
    fig.tight_layout()


# Define a function that plots a subplot
# for each datatype and hyperparameter to illustrate
# how the mini-batch val. RMSE changes for each value of the hyperparam.
def plot_performance_by_hyperparam(datatypes, horizons, min_or_mean='min'):
    """
    Function that plots the mini-batch validation RMSE per datatype, per horizon, 
    and per value of each different hyperparameter used to train the models.

    Parameters
    ----------
     datatypes : str or 1d-array
                {'images', 'images3d', 'irradiance', 'weather', 'combined'}
        - If str, plots RMSE by hyperparameter for this datatype.
        - If 1d-array, plots RMSE by hyperprameter for each
        of the specified datatypes.
    horizons : int or 1d-array
        - If int, the single specified forecast horizon will be plotted
        by hyperparmaeter value.
       - If 1d-array, all specified horizons will be plotted
       by hyperparameter value.
    min_or_mean : {'min', 'mean'} (Default: 'min')
        - If min, plots the minimum RMSE per hyperparameter and horizon.
        - If mean, plots the mean RMSE per hyperparameter and horizon.
    
    Returns
    -------
    None
        Only plots a graph.
    """

    # Change the type of some of the parameters to an array, if needed.
    if isinstance(datatypes, str):
        datatypes = [datatypes]
    if isinstance(horizons, int):
        horizons = [horizons]

    # Prepare figure and axes.
    fig, axs = plt.subplots(
        nrows=len(datatypes), 
        ncols=3, #one column for each hyperparameter
        figsize=(18, 5*len(datatypes)),
        sharey=True)
    fig.suptitle(f'{min_or_mean.title()} RMSE by Horizon, Hyperparameter, and Datatype\n',
                    fontsize=20)
    colors = cm.Set2.colors
    markers = ['o', 'v', 's']

    # Define a dictionary of the parameters that we used in the very
    # first training step, i.e. the params used to find the best
    # layer architecture per datatype:
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

    # Also prepare a dictionary that denotes which hyperparameters were used 
    # in the second step, i.e. to trainthe different models to find the 
    # optimal delta-length combination per forecast horizon.
    hyper_dict = copy.deepcopy(layer_dict)
    
    for datatype in ['irradiance', 'weather', 'combined']:
        
        # The only hyperparam we changed from the first to the second step
        # was to increase the learning rate for the LSTMs to ensure
        # faster convergence.
        hyper_dict[datatype]['lr'] = 3e-04

    # Iterate through each datatype and horizon.
    for i, datatype in enumerate(datatypes):

        # For each datatype, get the index of the layers architecture
        # that performed the best in the first step.
        layers_idx = getperformance.get_best_layers(
            datatype=datatype, 
            return_layer_params=False, 
            **layer_dict[datatype],
            output_missing=False)

        for j, horizon in enumerate(horizons):

            # For each datatype-horizon combination,
            # get the best performing delta-length combination.
            _, delta, length = getperformance.get_best_model_by_horizon(
                datatype=datatype, 
                horizon=horizon, 
                return_delta_length=True,
                return_hyperparams=False,
                batch_size=hyper_dict[datatype]['batch_size'], 
                lr=hyper_dict[datatype]['lr'], 
                weight_decay=hyper_dict[datatype]['weight_decay'])

            # Get all files containing the RMSEs for each hyperparameter combination
            # used to train this datatype-horizon-delta-length combination.
            loss_dir = f'./performance/{datatype}/{datatype}_layers_{layers_idx}/delta_{delta}/' \
                        f'length_{length}/y_{horizon}/'
            loss_dfs = [f for f in os.listdir(loss_dir) if not f.startswith('.')
                        and 'final' not in f] #ensures that the final model trained on
                                                #100% of the data is excluded here

            # Prepare the output that collects hyperparameters and RMSE.
            out = {'Batch Size':[],
                    'Learning Rate':[],
                    'Weight Decay':[],
                    'RMSE':[]}

            # Iterate through each file and extract the hyperparameters
            # and validation RMSEs.
            for loss_df in loss_dfs:

                # Add the hyperparameters and the RMSE to the output dictionary.
                hyperparams = getperformance.extract_hyperparameters(
                    hyperparam_name=loss_df.replace('.pkl', ''),
                    as_string=False)
                df = pd.read_pickle(loss_dir+loss_df)

                out['Batch Size'].append(hyperparams['batch_size'])
                out['Learning Rate'].append(hyperparams['lr'])
                out['Weight Decay'].append(hyperparams['weight_decay'])
                out['RMSE'].append(df['val'].min())
            
            # Create a dataframe from the output dictionary.
            # Then, create a grouped dataframe for each hyperparameter and plot
            # the RMSE for each grouped value.')
            df = pd.DataFrame.from_dict(out)

            for k, hyper in enumerate(['Batch Size', 'Learning Rate', 'Weight Decay']):

                # Either group the RMSEs per hyperparameter by the
                # min or mean RMSE.
                if min_or_mean == 'min':
                    df_hyper = df.groupby(by=[hyper]).min()[['RMSE']]
                elif min_or_mean == 'mean':
                    df_hyper = df.groupby(by=[hyper]).mean()[['RMSE']]
                df_hyper.reset_index(inplace=True)

                # Plot the RMSE by hyperparameter
                axs[i, k].plot(
                    df_hyper[hyper],
                    df_hyper['RMSE'],
                    color=colors[j],
                    marker=markers[j],
                    label=f'horizon: {horizon}')
                axs[i, k].tick_params(labelsize=14)

                # The hyperparams learning rate and weight decay
                # need a log scale with base 10.
                if hyper in ['Learning Rate', 'Weight Decay']:
                    axs[i, k].set_xscale('log')

                # The batch size hyperparameter needs a log scale
                # with base 2.
                if hyper == 'Batch Size':
                    axs[i, k].set_xscale('log', base=2)
                    axs[i, k].xaxis.set_major_formatter(ScalarFormatter())

                # Add labels depending on the position in the overall plot.
                if k == 0: #only add y labels on the very left, i.e. for the first hyperparams.
                    axs[i, k].set_ylabel(f'{min_or_mean.title()} Mini-Batch RMSE (W/m^2)', fontsize=14)
                 
                if i == len(datatypes)-1: #only add x labels to the very bottom
                    axs[i, k].set_xlabel(hyper, fontsize=14)

                if k == 1: #only add a title to the figure in the middle
                    axs[i, k].set_title(datatype, fontsize=16)

    # Final figure adjustments.
    plt.legend(loc='best', fontsize=16)
    fig.tight_layout()


# Define a function that plots the a histogram of the validation 
# mini-batch RMSEs for each specified datatype and horizon.
def plot_rmse_hist_by_horizon_of_hyperparams(datatypes, horizons):
    """
    Function that plots a subplot for each specified datatype. 
    Each subplot will contain one or multiple histogram(s) of the
    mini-batch validation RMSEs of the hyperparameter-tuning step 
    in the training process. Each subplot will have one histogram
    for every specified forecast horizon.

    Parameters
    ----------
    datatypes : str or 1d-array
                {'images', 'images3d', 'irradiance', 'weather', 'combined'}
        - If str, plots RMSE histograms of this particular datatype.
        - If 1d-array, plots RMSE histograms for each of the specified 
        datatypes.
    horizons : int or 1d-array
        - If int, plots RMSE histograms of just one forecast horizon
        per datatype.
        - If 1d-array, plots RMSE histograms of multiple horizons
        per datatype.
    
    Returns
    -------
    None
        Only plots a graph.
    """

    # Change the type of the parameters to an array, if needed.
    if isinstance(datatypes, str):
        datatypes = [datatypes]
    if isinstance(horizons, int):
        horizons = [horizons]

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

    # Prepare shape of the subplot grid. We'll use max. 3 columns
    # and calculate the number of rows from that.
    max_cols = 3

    if len(datatypes) >= max_cols:
        ncols = max_cols
    else:
        ncols = len(datatypes) 
    nrows = math.ceil(len(datatypes) / max_cols)
    
    # Prepare the figure.
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(ncols*7.5, nrows*6),
        sharex=True)
    fig.suptitle(f'Histogram of Mini-Batch Validation RMSEs Per Datatype and Forecast Horizon',
                fontsize=20)
    try:
        axs = axs.flatten()
    except AttributeError:
        axs = [axs] #needed if only one element in datatypes was given
    colors = cm.Set2.colors

    # Initialize a dictionary to save the data.
    out = {datatype:
            {horizon:[]
                for horizon in horizons}
            for datatype in datatypes}
    
    # Iterate through each datatype and get the best performing layer index.
    for datatype in datatypes:

        best_idx = getperformance.get_best_layers(
            datatype=datatype,
            return_layer_params=False,
            delta=param_dict[datatype]['delta'],
            length=param_dict[datatype]['length'],
            horizon=param_dict[datatype]['y'],
            batch_size=param_dict[datatype]['batch_size'],
            lr=param_dict[datatype]['lr'],
            weight_decay=param_dict[datatype]['weight_decay'],
            output_missing=False)

        # Iterate through each horizon and get the best performing delta-length combo.
        for horizon in horizons:

            _, best_delta, best_length, = getperformance.get_best_model_by_horizon(
                datatype=datatype, 
                horizon=horizon, return_delta_length=True, 
                return_hyperparams=False,
                batch_size=None,
                lr=None, 
                weight_decay=None) 

            # Create a string for the directory containing the RMSES of all the 
            # trained hyperparameters for this datatype, horizon, architecture,
            # and delta-length combo.
            rmse_dir = f'./performance/{datatype}/{datatype}_layers_{best_idx}/' \
                        f'delta_{best_delta}/length_{best_length}/y_{horizon}'

            # Read each one RMSE dataframe in this directory and
            # add their best validation RMSE to the output dictionary
            # for each of them.
            rmse_paths = [f'{rmse_dir}/{i}' for i in os.listdir(rmse_dir) if '.pkl' in i]
                        
            for rmse_path in rmse_paths:

                rmse_df = pd.read_pickle(rmse_path)
                rmse = rmse_df['val'].min()
                out[datatype][horizon].append(rmse)

    # Now that we have aggregated the data, we can plot it.
    # Iterate through each datatype and horizon and plot the respective 
    # histogram of validation RMSES.
    for i, datatype in enumerate(datatypes):
        
        for j, horizon in enumerate(horizons):

            # Extract the data and calculate number of bins.
            data = out[datatype][horizon]
            iqr = stats.iqr(data) #interquartile range
            bin_width = 2 * iqr * (len(data)**(-1/3)) #Freedman–Diaconis rule
            num_bins = (max(data)-min(data)) / bin_width

            # If more than one horizon is specified, the histograms
            # should be semi-transparent to make them easier
            # to read.
            if len(horizons) > 1:
                alpha = 0.5
            else:
                alpha = 1
            
            # Plot the histogram.
            axs[i].hist(
                x=data, 
                bins=int(num_bins),
                color=colors[j],
                label=f'y_{horizon}',
                alpha=alpha)
            axs[i].set_title(datatype, fontsize=16)
            axs[i].set_xlabel('Validation Mini-Batch RMSE (W/m^2)', fontsize=16)

            if i % max_cols == 0:
                axs[i].set_ylabel('Frequency', fontsize=16)
            axs[i].tick_params(labelsize=14)
   
    # Hide the remaining empty subplots.
    num_remaining = len(axs) - len(datatypes)
    for i in range(len(axs)-num_remaining, len(axs)):
        axs[i].axis('off')

    # Add the legend either on or next to a subplot.
    handles = [Rectangle((1,0),1,1,color=colors[i],ec="k") for i in range(len(horizons))]
    labels= [f'y_{horizon}' for horizon in horizons]

    if len(datatypes) > max_cols and len(datatypes) % max_cols != 0:
        plt.legend(handles, labels, 
        loc='center left', fontsize=16) #next to last subplot
    else:
        plt.legend(handles, labels, loc='best', fontsize=16)
    
    fig.tight_layout()


# Define a function that plots the validation (mini-batch) RMSE 
# of the specified datatype for each given forecast horizon.
def plot_rmse_by_horizon(datatypes, horizons, full_data=False,
                        include_ensemble=False):
    """
    Function that plots a linegraph of horizon vs. validation 
    (mini-batch RMSE) for each datatype into one large plot.

    Parameters
    ----------
    datatypes : str or 1d-array
                {'images', 'images3d', 'irradiance', 'weather', 'combined'}
        - If str, plots RMSEs of this particular datatype.
        - If 1d-array, plots RMSEs for each of the specified datatypes.
    horizons : int or 1d-array
        - If int, plots RMSE of just one forecast horizon per datatype.
        - If 1d-array, plots RMSE of multiple horizons per datatype.
    full_data : bool (Default: False)
        - If True, plots the validation RMSE of the whole validation set.
        - If False, plots the mean mini-batch val. RMSE that was computed 
        during training of each model.
    include_ensemble : bool (Default: False)
        - If True, also includes the linear and NN ensemble models
        - If False, only includes the individual base models.
    
    Returns
    -------
    None
        Only plots a graph.
    """

    print('Preparing the plot of forecast horizons vs validation RMSEs per datatype...')

    # Change the type of the parameters to an array, if needed.
    if isinstance(datatypes, str):
        datatypes = [datatypes]
    if isinstance(horizons, int):
        horizons = [horizons]

    # Initilaize figure and axis.
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title(
        f'Validation RMSE Per Datatype and Forecast Horizon',
        fontsize=20)
    ax.set_xlabel('Forecast horizon (minutes)', fontsize=16)

    if full_data == True:
        ax.set_ylabel('Validation RMSE (W/m^2)', fontsize=16)
    elif full_data == False:
        ax.set_ylabel('Mean Mini-Batch Validation RMSE (W/m^2)', fontsize=16)

    ax.tick_params(labelsize=14)
    colors = cm.Set1.colors
    markers = ['o', 'v', '^', 's', '*', 'd'] * 3

    # Initilaize the output where the data shall be stored.
    if include_ensemble == True:
        out = {datatype:
            {'RMSEs':[],
            'Horizons':[]}
            for datatype in datatypes+['persistence', 
                                        'linear ensemble', 
                                        'NN ensemble']}
    
    elif include_ensemble == False:
        out = {datatype:
                {'RMSEs':[],
                'Horizons':[]}
                for datatype in datatypes+['persistence']}

    # Define a dictionary of the parameters that we used in the very
    # first training step, i.e. the params used to find the best
    # layer architecture per datatype:
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
    
    # Iterate through each datatype and get the best performing 
    # layer architecture.
    if full_data == False: #utilize the mean mini-batch RMSEs

        for datatype in datatypes:

            layers_idx = getperformance.get_best_layers(
                datatype=datatype, 
                return_layer_params=False, 
                **layer_dict[datatype],
                output_missing=False)

            # For each datatype and the respective architecture, iterate through
            # each horizon and get the best performing delta-length-hyperparameter
            # combination.
            for horizon in horizons:

                # For each datatype-horizon combination,
                # get the best performing delta-length combination.
                _, delta, length, hyper = getperformance.get_best_model_by_horizon(
                    datatype=datatype, 
                    horizon=horizon, 
                    return_delta_length=True,
                    return_hyperparams=True,
                    batch_size=None, #search across all hyperparams for the best ones
                    lr=None,
                    weight_decay=None)

                # Create the correct filename based on the best performing hyperparameters.
                name = f'batch_size({hyper["batch_size"]})__lr({hyper["lr"]})__' \
                        f'weight_decay({hyper["weight_decay"]}).pkl'
                
                # Extract the validation rmse from the correct path based on the best-
                # performing delta-length-hyperparam combination.
                rmse_path = f'./performance/{datatype}/{datatype}_layers_{layers_idx}/' \
                            f'delta_{delta}/length_{length}/y_{horizon}/{name}'
                try:
                    rmse_df = pd.read_pickle(rmse_path)
                    rmse = rmse_df['val'].min()
                except FileNotFoundError:
                    rmse = None

                # Add the rmse and the corresponding horizon to the output.
                out[datatype]['RMSEs'].append(rmse)
                out[datatype]['Horizons'].append(horizon)

    # If we want the full-data validation RMSEs instead of the mean mini-batch val. RMSEs
    # computed during training, we can utilize the predictions made by each model on
    # the whole dataset.
    elif full_data == True:

        # Iterate through each horizon and get the ensemble data, i.e. the data constisting
        # of the true targets and the individual model predictions.
        for horizon in horizons:

            val_df = data.get_ensemble_data(
            horizon=horizon,
            only_return_val_df=True)

            # Calculate the RMSE on the whole dataset for each datatype
            # and add it to the output.
            for datatype in datatypes:

                errors = (val_df[datatype].to_numpy() - val_df['y'].to_numpy())**2
                rmse = np.sqrt(np.mean(errors))
                out[datatype]['RMSEs'].append(rmse)
                out[datatype]['Horizons'].append(horizon)

    # We also need the predictions of the naive persistence model. For this,
    # we first need the correct data for each horizon so we can feed it into 
    # the model.
    print('Extracting, preprocessing, and predicting on the irradiance data '
        'using the persistence model...')
    dataset = data.create_dataset(type='irradiance', from_pkl=True)
    
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
        (_, X_val, _,
        _, y_val, _, #we only need X and y validation data
        _, _, _) = preprocessing.split_and_preprocess(
            X=X, y=y, timestamps=t, preprocessing=None)

        # Make predictions on the validation set and add
        # the RMSE to the output.
        model = models.Persistence()
        yhat = model(X_val)
        yhat = yhat.detach().numpy().flatten()
        y_val = y_val.detach().numpy().flatten()
        errors = (y_val - yhat) ** 2

        # If we want to look at the mean mini-batch RMSEs computed during training,
        # we need to do the same for the persistence model.
        # To make it comparable, we also need to calculate the RMSE
        # of the persistence model on mini batches.
        if full_data == False:
            rmses = []
            for j in range(0, len(errors), 64): #we'll use a batch size of 64
                rmse_batch = np.sqrt(np.mean(errors[j:j + 64]))
                rmses.append(rmse_batch)
            rmse = np.mean(rmses)

        # Otherwise, we can compute the full-data RMSEs.
        elif full_data == True:
            rmse = np.sqrt(np.mean(errors))

        # Add it to the output.
        out['persistence']['RMSEs'].append(rmse)
        out['persistence']['Horizons'].append(horizon)

    # Lastly, if specified, we also want the RMSEs of the ensemble models.
    if include_ensemble == True:

        # First, get the full validation RMSEs for all linear ensemble models
        # for every possible ensemble input combination
        linear_df = getperformance.compare_linear_combinations(
            horizons=horizons)

        # Extract the minimum RMSE at each horizon
        for horizon in horizons:
            rmse = linear_df.loc[:, (f'y_{horizon}', 'Val RMSE')].min()
            
            # Add that RMSE and the horizon to the output.
            out['linear ensemble']['RMSEs'].append(rmse)
            out['linear ensemble']['Horizons'].append(horizon)

        # Now, we need the full validation RMSEs for each of the best-
        # performing NN ensemble model per horizon. For this, we need
        # the best performing model, input, and hyperparameter combos:

        # Get the best performing inputs and hyperparams for the intra-hour horizons.
        model_y30, ensemble_inputs_y30, ensemble_hyper_y30 = getperformance.get_best_ensemble_model_by_horizon(
        horizon=30, return_inputs=True, return_hyperparams=True, #return hyperparams
        batch_size=None, lr=None, weight_decay=None) #search across all hyperparams

        # Get the best performing inputs and hyperparams for the intra-day horizons.
        model_y120, ensemble_inputs_y120, ensemble_hyper_y120 = getperformance.get_best_ensemble_model_by_horizon(
            horizon=120, return_inputs=True, return_hyperparams=True, #return hyperparams
            batch_size=None, lr=None, weight_decay=None) #search across all hyperparams

        # Get the best performing inputs and hyperparams for the day-ahead horizons.
        model_y1440, ensemble_inputs_y1440, ensemble_hyper_y1440 = getperformance.get_best_ensemble_model_by_horizon(
            horizon=1440, return_inputs=True, return_hyperparams=True, #return hyperparams
            batch_size=None, lr=None, weight_decay=None) #search across all hyperparams

        # Also get the index of the best performing layer architecture.
        best_idx = getperformance.get_best_ensemble_layers(
            batch_size=128, #we used these hyperparams to find the best architecture 
            lr=0.0001,     
            weight_decay=1e-05,
            return_idx=True)

        # Define which horizons belong to which horizon bucket.
        intra_hours = [i for i in horizons if i < 60] #everything < 1 hour
        intra_days = [i for i in horizons if i >=60 and i < 24*60] #everything < 1 day
        day_aheads = [i for i in horizons if i >= 24*60] #everything > 1 day

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
            _, X_val, _, _, y_val, _, _, _, _, _ = data.get_ensemble_data(
                horizon=horizon,
                **inputs)

            # Generate the directory containing all models of the best layer architecture
            # best input combination with the model weights of the hyperparameter combinations.
            hyper_dir = f'./parameters/ensemble/ensemble_layers_{best_idx}/{input_string}/y_{horizon}'
            hyper_file = f'batch_size({hyperparams["batch_size"]})__lr({hyperparams["lr"]})__' \
                        f'weight_decay({hyperparams["weight_decay"]})__early_stopping(True).pt'
            
            # Load in the weights into the model.
            model_path = f'{hyper_dir}/{hyper_file}'
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            model = model.eval()

            # Calculate the full validation RMSE.
            y_hat = model(torch.from_numpy(X_val).float())
            y_hat = np.array(y_hat.squeeze(1).detach())
            errors = (y_hat - y_val) ** 2
            rmse = np.sqrt(np.mean(errors))

            # Add that RMSE and the horizon to the output.
            out['NN ensemble']['RMSEs'].append(rmse)
            out['NN ensemble']['Horizons'].append(horizon)

    # Initialize the maximum RMSE; needed for text placement.
    max_rmse = 0
    
    # Now that we aggregated the data, we can plot them.
    for i, datatype in enumerate(out.keys()):
        
        # Set the colors.
        if datatype == 'persistence':
            color = 'black'
        else:
            color = colors[i]

        # Check whether the max RMSE of this datatype is the current maximum.
        # If yes, change it.
        if max_rmse < max(out[datatype]['RMSEs']):
            max_rmse = max(out[datatype]['RMSEs'])
        
        # Plot the data.
        ax.plot(
            [str(i) for i in out[datatype]['Horizons']], #makes plot more readable
            out[datatype]['RMSEs'],
            marker=markers[i],
            color=color,
            label=datatype)

    # We also want to plot vertical lines to indicate where the 
    # different forecast horizon buckets start. For this, we 
    # need the position of the buckets in the specified horizons list.
    intra_hour = [i<60 for i in horizons]
    intra_day = [i<60*24 for i in horizons]

    # We need to find the last horizon within each bucket. For this,
    # reversing the lists can help.
    intra_hour = intra_hour[::-1]
    intra_day = intra_day[::-1]

    # Now we can find the last instance within each bucket
    last_intra_hour = len(intra_hour) - 1 - intra_hour.index(True)
    last_intra_day = len(intra_day) - 1 - intra_day.index(True)
    
    # Plot vertical lines to indicate the forecast horizon buckets.
    ax.axvline(
        x=last_intra_hour+0.5, #line between intra-hour and intra-day
        linestyle='--',
        dashes=(5, 10),
        color='dimgrey')
    ax.axvline(
        x=last_intra_day+0.5, #line between intra-day and day-ahead
        linestyle='--',
        dashes=(5, 10),
        color='dimgrey')

    # Add labels to the lines.
    ax.text(
        x=last_intra_hour+0.45,
        y=max_rmse,
        s='<- intra-hour',
        color='black',
        horizontalalignment='right',
        fontsize='large')
    ax.text(
        x=last_intra_day+0.55,
        y=max_rmse,
        s='day-ahead ->',
        color='black',
        horizontalalignment='left',
        fontsize='large')
        
    # Final enhancement to figure.
    plt.legend(loc='upper right', fontsize=16)
    fig.tight_layout()


# Define a function that plots a scatterplot of the full-data
# validation errors vs. a given metric, such as hour of the day
# by datatype
def plot_errors_by_metric(metric, datatypes, sample_size, horizons, which_set='val'):
    """
    Function that first finds aligns the prediction RMSEs of each specified
    datatype and horizon on the specified train/val/test set
    with the specified metric (e.g. time of day.). 
    These errors are then plotted as a scatterplot vs. the given metric.
    
    Parameters
    ----------
    metric : {'time', 'month', 'irradiance', 'air_temp', 'relhum', 
                'press', 'windsp', 'winddir', 'max_windsp'}
        What metric to plot the validation errors against.
    datatypes : str or 1d-array
                {'images', 'images3d', 'irradiance', 'weather', 'combined'}
        - If str, plots errors vs. metric of this particular datatype.
        - If 1d-array, plots errors vs. metric for each of the specified 
        datatypes.
    sample_size : float or str
        Whether to get the errors by predicting on the whole validation set
        or on just a sample of it to speed up the process.
        - If float, gets the sample of the data according to a fixed fraction,
        of the dataset size, e.g. 0.1 of all data
        - If str, gets the sample according to a certain frequency between
        timestamps, e.g. '3min' for 3 minute deltas.
        See https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
        for all accepted frequencies.
    horizons : int or 1d-array
        - If int, plots the errors vs. metric of just one forecast horizon
        per datatype.
        - If 1d-array, plots the errors vs. metric of multiple horizons
        per datatype.
    which_set : {'train', 'val', 'test'} (Default: 'val)
        Whether to get the predictions on the train, val, or test set.

    Returns
    -------
    None:
        Only plots a graph
    """

    # Change the type of some of the parameters to an array, if needed.
    if isinstance(datatypes, str):
        datatypes = [datatypes]
    if isinstance(horizons, int):
        horizons = [horizons]

    # Aggregate errors by datatype and horizon and align them with
    # specified metric.
    error_dict = align_errors_by_metric(
        metric=metric,
        sample_size=sample_size,
        datatypes=datatypes,
        horizons=horizons,
        which_set=which_set)
    
    # Prepare shape of the subplot grid. We'll use max. 3 columns
    # and calculate the number of rows from that.
    max_cols = 3

    num_models = len(error_dict.keys())
    if num_models >= max_cols:
        ncols = max_cols
    else:
        ncols = num_models
    nrows = math.ceil(num_models / max_cols)

    # Prepare figure titles and axes
    if which_set == 'train':
        title = 'Training'
    elif which_set == 'val':
        title = 'Validation'
    elif which_set == 'test':
        title = 'Test'
    
    # Prepare figure and axes.
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(ncols*7.5, nrows*6),
        sharey=True)
    fig.suptitle(f'{title} Errors vs. {metric.title()} by Datatype and Horizon',
                fontsize=20)
    axs = axs.flatten()
    colors = cm.Set2.colors
    xlabel_dict = {'time':'Time of the day',
                    'month':'Month of the year',
                    'irradiance':'GHI (W/m^2)',
                    'air_temp':'Ambient temperature (°C)',
                    'relhum':'Relative humidity (%)',
                    'press':'Pressure (Pa)',
                    'windsp':'Wind speed (m/s)',
                    'winddir':'Wind direction',
                    'max_windsp':'Maximum wind speed (m/s)'}

    # Iterate through datatypes and horizons and plot
    # the respective scatterplots onto the correct axes.
    for i, datatype in enumerate(error_dict.keys()):

        for j, horizon in enumerate(horizons):

            # Plot the data.
            axs[i].scatter(
                x=error_dict[datatype][horizon][1], #plot the metric
                y=error_dict[datatype][horizon][0], #plot the errors
                color=colors[j],
                label=f'y_{horizon}',
                alpha=0.4,
                s=100)

            # Also add some formatting to the plot.
            axs[i].axhline(
                y=0, #indicate where y_hat == y_true
                linestyle='--',
                dashes=(5, 10),
                color='dimgrey')
            axs[i].tick_params(labelsize=14)
            axs[i].set_title(datatype, fontsize=16)
            axs[i].set_xlabel(xlabel_dict[metric], fontsize=14)
            if i % max_cols == 0:
                axs[i].set_ylabel(f'{title} Forecast Errors (W/m^2)', fontsize=14)

            # For the time-based metrics, we need to adjust the x-axis.
            if metric == 'time':
                axs[i].xaxis.set_major_formatter(DateFormatter('%H:%M'))
            elif metric == 'month':
                axs[i].xaxis.set_major_formatter(DateFormatter('%m'))

    # Hide the remaining empty subplots.
    num_remaining = len(axs) - num_models
    for i in range(len(axs)-num_remaining, len(axs)):
        axs[i].axis('off')
    
    # Final figure adjustments.
    plt.legend(loc='best', fontsize=16)
    fig.tight_layout()

         
# Define a function that aligns validation RSMEs to a specified metric
# for different models.
def align_errors_by_metric(metric, sample_size, datatypes, horizons, which_set):
    """
    Function that creates a dictionary that contains the errors 
    aligned with a specified metric for each available time stamp
    within the train/val/test set for all specified datatypes.
    
    Parameters
    ----------
    metric : {'time', 'month', 'irradiance', 'air_temp', 'relhum', 
                'press', 'windsp', 'winddir', 'max_windsp'}
        By what metric to aggregate the RSMEs.
    sample_size : float
        Whether to get the errors of the predictions on the whole set
        or on just a sample of it, e.g. 0.01 for 1% of the data.
    datatypes : str or 1d-array
                {'images', 'images3d', 'irradiance', 'weather', 'combined'}
        - If str, aggregate the RSMEs for this datatype.
        - If 1d-array, aggregates the RSMEs for each of
        the given datatypes.
    horizons : int or 1d-array
        If int, aggregates the errors for only one forecast horizon.
        If 1d-array, aggregates the errors for multiple horizons.
    which_set : {'train', 'val', 'test'}
        Whether to get the predictions on the train, val, or test set.

    Returns
    ------
    error_dict : dict
        {'datatype1':{'horizon1':(metrics, errors), 'horizon2':(metrics, errors)}, ...}
        A dictionary containing keys of datatypes incl. the 'persistence' model
        and values of tuples with the RMSEs aligned with the specified metric.
    """

    # Change the type of some of the parameters to an array, if needed.
    if isinstance(datatypes, str):
        datatypes = [datatypes]
    if isinstance(horizons, int):
        horizons = [horizons]

    # Get a series of the relevant metrics for each timestamp
    # in order to align validation errors to the metric.
    metric_df = data.create_dataset(type='weather', from_pkl=True).data
    if metric == 'time':
        metric_df['metric'] = [i.replace(year=2015, month=1, day=1) #we only care
                                for i in metric_df.index] #about the time here
    elif metric == 'month':
        metric_df['metric'] = metric_df.index
    elif metric == 'irradiance':
        metric_df.rename(columns={'ghi':'metric'}, inplace=True)
    elif metric in ['air_temp', 'relhum', 'press', 'windsp', 'winddir', 'max_windsp']:
        metric_df.rename(columns={metric:'metric'}, inplace=True) 
    else:
        raise ValueError("metric argument must be one of {'time', 'month'," 
            "'irradiance', 'air_temp', 'relhum',  'press', 'windsp', 'winddir', 'max_windsp'}")       
    
    metric_df = metric_df[['metric']]

    # Prepare a dictionary that collects all predictions for each horizon and datatype
    # and the timestamps of these predictions. It also collects all true targets y
    # and their timestamps.
    data_dict = {datatype:{horizon:{'data':None, 'time':None} for horizon in horizons}
                for datatype in datatypes+['targets', 'persistence']}

    for datatype, horizon in itertools.product(datatypes, horizons):

        # Load the predictions on the correct dataset and their timestamps
        # depending on the specified datatype and forecast horizon.
        yhat_path = f'./datasets/data_for_analysis/predictions/predictions{horizon}_' \
                    f'{which_set}_{datatype}.npy'
        time_path = f'./datasets/data_for_analysis/predictions/predictions{horizon}_' \
                    f'{which_set}_{datatype}_time.npy'

        yhat = np.load(yhat_path)
        yhat_t = np.load(time_path)

        # Also load in the correct targets and their timestamps based on the specified horizon.
        y_path = f'./datasets/data_for_analysis/targets/y_{horizon}.npy'
        y_time_path = f'./datasets/data_for_analysis/targets/y_{horizon}_time.npy'

        y = np.load(y_path)
        y_t = np.load(y_time_path)

        # Convert the targets to the correct timezone, i.e. from UTC to PST. 
        # Note that the yhats are alreadyin the correct timezone as the models were
        # fed input with the correct timezone.
        pst = pytz.timezone('America/Los_Angeles')
        y_t = pd.to_datetime(y_t).tz_localize(pytz.utc).tz_convert(pst)
        y_t = y_t.tz_localize(None)
        y_t = np.array(y_t)

        # Save the predictions, their timestamps, the targets, and their timestamps
        # all to the dictionary.
        data_dict[datatype][horizon]['data'] = yhat
        data_dict[datatype][horizon]['time'] = yhat_t
        data_dict['targets'][horizon]['data'] = y
        data_dict['targets'][horizon]['time'] = y_t

    # We also need the predictions of the naive persistence model. For this,
    # we first need the correct data for each horizon so we can feed it into 
    # the model.
    print(f'Extracting, preprocessing, and predicting on the {which_set} dataset '
        'using the persistence model...')
    dataset = data.create_dataset(type='irradiance', from_pkl=True)
    
    for horizon in horizons:
        X, y, t = dataset.get_data(
            datatype='irradiance',
            forecast_horizon=horizon,
            sample=False,
            sample_size=1,
            delta=1,
            length=10)

        # Split the data into train, val, and test sets:
        (X_train, X_val, X_test,
        _, _, _, #the true targets; not needed in this step
        t_train, t_val, t_test) = preprocessing.split_and_preprocess(
            X=X, y=y, timestamps=t, preprocessing=None)

        # Make predictions on one of these sets and add
        # the predictions and their timestamp to the dictionary.
        model = models.Persistence()

        if which_set == 'train':
            yhat = model(X_train)
            yhat = yhat.detach().numpy().flatten()
            data_dict['persistence'][horizon]['data'] = yhat
            data_dict['persistence'][horizon]['time'] = t_train
        
        elif which_set == 'val':
            yhat = model(X_val)
            yhat = yhat.detach().numpy().flatten()
            data_dict['persistence'][horizon]['data'] = yhat
            data_dict['persistence'][horizon]['time'] = t_val
       
        elif which_set == 'test':
            yhat = model(X_test)
            yhat = yhat.detach().numpy().flatten()
            data_dict['persistence'][horizon]['data'] = yhat
            data_dict['persistence'][horizon]['time'] = t_test
    
    # We can now start calculating the errors for each datatype and horizon.
    print('Calculating errors for each datatype and horizon and aligning '
        'each error with the specified metric for the scatterplot...')
    
    # First, we need to find the common timestamps among all these data.
    # For this, we can utilize the index system of a pandas dataframe.
    # This method is a lot faster than searching for the intersection across
    # all timestamp sets.
    # Let's first create a dataframe with the timestamps of the first datatype-horizon
    # combinatin as an index.
    timestamps_1 = data_dict[datatypes[0]][horizons[0]]['time']
    time_df1 = pd.DataFrame(
        data=[0 for i in range(len(timestamps_1))],
        index=timestamps_1,
        columns=['0'])

    # We now iterate through all timestamps and create a similar df for them.
    # Merging these df's with an 'inner' join will result in the index
    # being the intersection of the timestamps.
    for i, (datatype, horizon) in enumerate(
        itertools.product(datatypes+['targets', 'persistence'], horizons)):

        # We can skip the first timestamp as those are in time_df1.
        if i != 0:
            
            # Extract the timestamps for this datatype-horizon combination.
            timestamps_2 = data_dict[datatype][horizon]['time']

            # Build the df and merge the 2 df's.
            time_df2 = pd.DataFrame(
                data=[0 for i in range(len(timestamps_2))],
                index=timestamps_2,
                columns=[str(i+1)])
            time_df1 = time_df1.merge(
                right=time_df2,
                left_index=True,
                right_index=True,
                how='inner')

    # The resulting merged dataframe will have the intersection
    # of all timestamps as an index.
    intersection = time_df1.index.to_numpy()

    # If specified, we only want to plot a sample of these timestamps
    # or rather a sample of the predictions. This is advised, as otherwise
    # the scatterplot will be too crowded.
    if sample_size < 1:
        intersection = list(intersection)
        sample_num = int(len(intersection) * sample_size)
        random.seed(42)
        sample = random.sample(intersection, sample_num)

    else:
        sample = list(intersection)

    # We can now create the dictionary containing the errors and their timestamps
    # for each of the datatype-horizon combinations.
    # Let's initialize the dictionary where we store each error array.
    error_dict = {datatype:{horizon:() for horizon in horizons} 
                    for datatype in datatypes+['persistence']}

    # For each datatype-horizon combination, get the corresponding
    # true y targets and calculate the error for each prediction
    # whose timestamp is included in the sample.
    for datatype, horizon in itertools.product(datatypes+['persistence'], horizons):
        
        # Extract the correct data from the data dictionary.
        yhat = data_dict[datatype][horizon]['data']
        yhat_t = data_dict[datatype][horizon]['time']
        y = data_dict['targets'][horizon]['data']
        y_t = data_dict['targets'][horizon]['time']

        # We now need to filter these predictions and targets based on the 
        # common sample of timestamps. For this, we can again use a pandas 
        # dataframe for quick boolean filtering.
        yhat_df = pd.DataFrame(
            data=[False for i in range(len(yhat_t))],
			index=yhat_t,
			columns=['filter'])
        y_df = pd.DataFrame(
            data=[False for i in range(len(y_t))],
			index=y_t,
			columns=['filter'])

        # We now turn every False in this df into a True for every timestamp
        # among yhat_t and y_t that also exists in the common sample.
        yhat_df.loc[sample, 'filter'] = [True for i in range(len(sample))]
        y_df.loc[sample, 'filter'] = [True for i in range(len(sample))]

        # The 'filter' column can now be used as a boolean filter.
        yhat_filter = yhat_df['filter'].to_numpy()
        y_filter = y_df['filter'].to_numpy()

        # Apply these boolean filters to the predictions and the true targets.
        yhat = yhat[yhat_filter]
        y = y[y_filter]

        # We can now calculate the errors using these sampled predictions
        # and these sampled true targets
        errors = y - yhat

        # Almost done! We now need to align the metrics to the errors.
        # For this, we can join the df containing the filter for the 
        # predictions yhat with the df containing the metrics.
        yhat_df = yhat_df.merge(
            right=metric_df,
            how='left',
            left_index=True,
            right_index=True)

        # We can now extract the metric from that dataframe and filter
        # it using the same boolean filter we applied to the predictions.
        metrics = yhat_df['metric'].to_numpy()
        metrics = metrics[yhat_filter]

        # Done! We can now add the errors and their corresponding metrics
        # to the error dictionary at the correct keys corresponding to the
        # datatype-horizon combination.
        error_dict[datatype][horizon]
        error_dict[datatype][horizon] = (errors, metrics)

    return error_dict

    
# Define a function that plots the best linear combination models
# vs. the best individual models vs. the persistence model per horizon.
def plot_linear_combination_vs_indiv(horizons):
    """
    Function that first aggregates the best validation RMSEs of the individual models
    per forecast horizon. It then aggregates the same for the persistence model.
    Then, it fits all different linear combination models with the different
    ensemble data combinations and aggregates the best RMSEs of these per
    forecast horizon. These RMSEs are then plotted for each horizon.

    Parameters
    ----------
    horizons : int or 1d-array
        - If int, plots RMSE of just one forecast horizon per model type.
        - If 1d-array, plots RMSE of multiple horizons per model type.
    
    Returns
    -------
    None
        Only plots a graph.
    """

    # Change the type of the parameters to an array, if needed.
    if isinstance(horizons, int):
        horizons = [horizons]

    # Initilaize figure and axis.
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title(
        f'Validation RMSE Per Forecast Horizon Of The Different Models',
        fontsize=20)
    ax.set_xlabel('Forecast horizon (minutes)', fontsize=16)
    ax.set_ylabel('Validation RMSE (W/m^2)', fontsize=16)
    ax.tick_params(labelsize=14)
    colors = cm.Set1.colors
    markers = ['o', 'v', 'd']

    # We also want to plot vertical lines to indicate where the 
    # different forecast horizon buckets start. For this, we 
    # need the position of the buckets in the specified horizons list.
    intra_hour = [i<60 for i in horizons]
    intra_day = [i<60*24 for i in horizons]

    # We need to find the last horizon within each bucket. For this,
    # reversing the lists can help.
    intra_hour = intra_hour[::-1]
    intra_day = intra_day[::-1]

    # Now we can find the last instance within each bucket
    last_intra_hour = len(intra_hour) - 1 - intra_hour.index(True)
    last_intra_day = len(intra_day) - 1 - intra_day.index(True)

    # Calculate the RMSE for each indidividual model prediction per horizon.
    # Then extract the minimum RMSE for each horizon. Also extract the RMSEs
    # of the persistence model
    all_datatypes = ['images', 'images3d', 'irradiance', 'weather', 'combined']
    rmses_indiv_df = getperformance.compare_final_rmses(
        datatypes=all_datatypes,
        horizons=horizons)
    rmses_indiv = [rmses_indiv_df.loc[:, (f'y_{horizon}', 'RMSE')][:-1].min() 
                    for horizon in horizons]
    rmses_persistence = [rmses_indiv_df.loc['persistence', (f'y_{horizon}', 'RMSE')]
                    for horizon in horizons]

    # Calculate the RMSE for each ensemble data combination per horizon.
    # Then extract the minimum RMSE for each horizon.
    rmses_ensemble_df = df = getperformance.compare_linear_combinations(
        horizons=horizons)
    rmses_ensemble = [rmses_ensemble_df.loc[:, (f'y_{horizon}', 'Val RMSE')].min() 
                    for horizon in horizons]

    # Plot the best individual models RMSEs.
    ax.plot(
        [str(i) for i in horizons], #makes x-axis more readable
        rmses_indiv, #y-axis, i.e. RMSEs
        color=colors[0],
        marker=markers[0],
        label='Best Individual Model')

    # Plot the best linear combinations RMSEs.
    ax.plot(
        [str(i) for i in horizons], #makes x-axis more readable
        rmses_ensemble,
        color=colors[1],
        marker=markers[1],
        label='Best Linear Combination Model')

    # Plot the persistence model RMSEs.
    ax.plot(
        [str(i) for i in horizons], #makes x-axis more readable
        rmses_persistence, 
        color='black',
        marker=markers[2],
        label='Persistence Model')

    # Plot vertical lines to indicate the forecast horizon buckets.
    ax.axvline(
        x=last_intra_hour+0.5, #line between intra-hour and intra-day
        linestyle='--',
        dashes=(5, 10),
        color='dimgrey')
    ax.axvline(
        x=last_intra_day+0.5, #line between intra-day and day-ahead
        linestyle='--',
        dashes=(5, 10),
        color='dimgrey')
    
    # Add labels to the lines.
    ax.text(
        x=last_intra_hour+0.45,
        y=0+30,
        s='<- intra-hour',
        color='black',
        horizontalalignment='right',
        fontsize='large')
    ax.text(
        x=last_intra_day+0.55,
        y=0+30,
        s='day-ahead ->',
        color='black',
        horizontalalignment='left',
        fontsize='large')

    # Final enhancement to figure.
    plt.legend(loc='upper right', fontsize=16)
    fig.tight_layout()


# Define a function that plots the linear coefficients of the linear ensemble model.
def plot_linear_coef(horizons):
    """
    Function that fits a linear regression for a multitude of ensemble data
    combinations (e.g. with include_time=True) to account for increasingly
    more endogeneity. It then plots the linear weights of the individual
    model predictions for each of these data combinations into individual
    subplots across the specified time horizons.

    Parameters
    ----------
    horizons : int or 1d-array
        - If int, each subplot will only consist of one horizon on the x-axis.
        - If 1d-array, each subplot will contain multiple horizons on the x-axis.

    Returns
    -------
    None
        Only plots a graph.
    """

    # Change the type of horizons to an array, if needed.
    if isinstance(horizons, int):
        horizons = [horizons]

    # Initilaize figure and axis.
    fig, axs = plt.subplots(3, 2, figsize=(19, 13), sharey=True)
    fig.suptitle(
        f'Coefficients of the Linear Combination Model per Horizon',
        fontsize=20)
    markers = ['o', 'v', '^', 's', '*', 'd'] * 10
    colors = cm.Set1.colors
    axs = axs.flatten()

    # We also want to plot vertical lines to indicate where the 
    # different forecast horizon buckets start. For this, we 
    # need the position of the buckets in the specified horizons list.
    intra_hour = [i<60 for i in horizons]
    intra_day = [i<60*24 for i in horizons]

    # We need to find the last horizon within each bucket. For this,
    # reversing the lists can help.
    intra_hour = intra_hour[::-1]
    intra_day = intra_day[::-1]

    # Now we can find the last instance within each bucket
    last_intra_hour = len(intra_hour) - 1 - intra_hour.index(True)
    last_intra_day = len(intra_day) - 1 - intra_day.index(True)

    # Create a list of all ensemble input combinations we want to look at.
    # These will account for increasingly more endogeneity.
    data_list = [
        'predictions__',
        'predictions__interactions',
        'predictions__interactions__time',
        'predictions__interactions__time__hour_poly',
        'predictions__interactions__time__hour_poly__ghi',
        'predictions__interactions__time__hour_poly__ghi__weather']

    # Iterate through each ensemble input combo and extract the info
    # needed to generate that particular data.
    for i, data in enumerate(data_list):

        which_data = models.extract_ensemble_inputs(data)

        # Get a dataframe with the linear coefficients per horizon.
        coef_df = getperformance.compare_linear_coef(
            horizons=horizons,
            **which_data) #e.g. include_time=True

        # Filter df to only include coefficients of indiv. predictions.
        pred_only = ['images', 'images3d', 'irradiance', 'weather', 'combined']
        coef_df = coef_df[coef_df.index.isin(pred_only)]

        # Initialize the maximum weight (for text placement).
        max_weight = -1000000

        # Change the title of each subplot.
        if data == 'predictions__':
            title = 'predictions'
        else:
            title = data.replace('__', '+')

        # Plot the data by iterating through each coefficient.
        for j, coef in enumerate(coef_df.index):

            axs[i].plot(
            [str(k) for k in horizons], #makes x-axis more readable
            coef_df.loc[coef, :].to_numpy(),
            color=colors[j],
            marker=markers[j],
            label=coef)
            axs[i].set_title(title, fontsize=16)

            # Adust the maximum weight
            if coef_df.loc[coef, :].max() > max_weight:
                max_weight = coef_df.loc[coef, :].max()

        # Add the labels to the axes.
        if i > 3:
            axs[i].set_xlabel('Forecast horizon (minutes)', fontsize=16)
        if i % 2 == 1:
            axs[i].set_ylabel('Coefficient weights', fontsize=16)
        axs[i].tick_params(labelsize=14)

        # Plot vertical lines to indicate the forecast horizon buckets.
        axs[i].axvline(
            x=last_intra_hour+0.5, #line between intra-hour and intra-day
            linestyle='--',
            dashes=(5, 10),
            color='dimgrey')
        axs[i].axvline(
            x=last_intra_day+0.5, #line between intra-day and day-ahead
            linestyle='--',
            dashes=(5, 10),
            color='dimgrey')

        # Plot a horizontal line at 0.
        axs[i].axhline(
            y=0, #indicate where coefficients are not important
            linestyle='--',
            dashes=(5, 10),
            color='dimgrey')

        # Add labels to the lines.
        axs[i].text(
            x=last_intra_hour+0.45,
            y=max_weight,
            s='<- intra-hour',
            color='black',
            horizontalalignment='right',
            fontsize='large')
        axs[i].text(
            x=last_intra_day+0.55,
            y=max_weight,
            s='day-ahead ->',
            color='black',
            horizontalalignment='left',
            fontsize='large')

    # Final enhancement to figure.
    plt.legend(loc='best', fontsize=16)
    fig.tight_layout()


# Define a function that plots a histogram of the validation 
# RMSEs of all NN ensemble layer architectures.
def plot_ensemble_layers_hist(horizon, batch_size, lr, weight_decay):
    """
    Function that calculates the full validation RMSEs for all
    layer architectures of the NN ensemble model 
    for the specified horizon and hyperparams.
    It then creates a histogram of all these RMSEs.

    Parameters
    ----------
    horizon : int
        What forecast horizon the models were trained on.
    batch_size : int
        What batch sizes were used to train the models.
    lr : float
        What learning rates were used to train the models.
    weight_decay : float
        What weight decay was used to train the models.

    Returns
    -------
    None
        Only plots a graph.
    """

    # Initialize the RMSE list.
    rmses = []
    
    # Prepare the figure.
    fig, ax = plt.subplots(nrows=1, ncols=1,
                            figsize=(7.5, 6))
    fig.suptitle(f'Full Validation RMSEs Of The Ensemble Layer Architectures',
                fontsize=20)
    
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
        model_path = f'{filepath}{layer}/predictions__/y_{horizon}/{filename}'

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

            # Calculate the full validation RMSE and add it to the output.
            y_hat = model(torch.from_numpy(X_val))
            y_hat = np.array(y_hat.squeeze(1).detach())
            errors = (y_hat - y_val) ** 2
            rmse = np.sqrt(np.mean(errors))
            rmses.append(rmse)

        except FileNotFoundError:
            pass

    # Plot the histogram.
    iqr = stats.iqr(np.array(rmses)) #interquartile range
    bin_width = 2 * iqr * (len(rmses)**(-1/3)) #Freedman–Diaconis rule
    num_bins = (max(rmses)-min(rmses)) / bin_width
    ax.hist(x=rmses, bins=int(num_bins))
    ax.set_xlabel('Validation RMSE (W/m^2)', fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)
    ax.tick_params(labelsize=14)
    

# Define a function that plots a histogram of the full validation RMSE
# %-difference between the linear model and the NN ensemble model
# for each horizon.
def plot_linear_vs_nn_hist(horizons, batch_size, lr, weight_decay):
    """
    Function that utlizes the getperformance.compare_linear_vs_nn()
    function to aggregate the full validation RMSE data of the 
    linear and NN ensemble models for each ensemble data combination.
    It then plots a histogram for each specified horizon of the 
    %-difference in val. RMSEs between the two models.

    Parameters
    ----------
    horizons : int or 1d-array
        - If int, plot will only consists of one histogram.
        - If 1d-array, plot will contain one histogram per horizon.
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
    None
        Only plots a graph.
    """

    # Change horizons to an array, if needed.
    if isinstance(horizons, int):
        horizons = [horizons]

    # Aggregate the %-difference for all ensemble data combinations for each horizon
    # between the linear and NN ensemble models.
    linear_vs_nn = getperformance.compare_linear_vs_nn(
        horizons=horizons,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay)

    # Prepare shape of the subplot grid. We'll use max. 3 columns
    # and calculate the number of rows from that.
    max_cols = 3

    if len(horizons) >= max_cols:
        ncols = max_cols
    else:
        ncols = len(horizons) 
    nrows = math.ceil(len(horizons) / max_cols)
    
    # Prepare the figure.
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(ncols*7.5, nrows*6),
        sharex=True)
    fig.suptitle(
        f'Histogram of %-Difference of Validation RMSEs of the '
        'Linear vs. NN Ensemble Models Per Forecast Horizon',
        fontsize=20)
    try:
        axs = axs.flatten()
    except AttributeError:
        axs = [axs] #needed if only one element in datatypes was given
    colors = cm.Set2.colors

    # Iterate through the horizons and plot the histograms.
    for i, horizon in enumerate(horizons):

        # Extract the data and calculate number of bins.
        data = linear_vs_nn.loc[:, (f'y_{horizon}', '% vs. linear')].to_numpy()
        iqr = stats.iqr(data) #interquartile range
        bin_width = 2 * iqr * (len(data)**(-1/3)) #Freedman–Diaconis rule
        num_bins = (max(data)-min(data)) / bin_width
        num_bins = 10
        
        # Plot the histogram.
        axs[i].hist(
            x=data, 
            bins=int(num_bins),
            color=colors[i],
            label=f'y_{horizon}')
        axs[i].set_xlabel('%-Difference of Validation RMSEs', fontsize=16)
        axs[i].tick_params(labelsize=14)
        axs[i].set_title(f'y_{horizon}')

        if i % max_cols == 0:
                axs[i].set_ylabel('Frequency', fontsize=16)
        axs[i].tick_params(labelsize=14)

    # Final enhancement to figure.
    fig.tight_layout()


# Define a function that plots the %-difference in train, val, and test
# RMSEs between the best indiv. model and ensemble model across horizons.
def plot_train_val_test_difference_by_horizon(horizons):
    """
    Function that uses the getperformance.get_overall_train_val_test_errors()
    function to aggregate the train, val, and test RMSEs for every model
    and every forecast horizon. It then calculates the %-difference between the
    RMSE of the best individual model (including persistence baseline) and the
    best ensemble model (including linear baseline) for every specified horizon.

    Parameters
    ----------
    horizon : 1d-array
        Over which forecast horizons to plot the %-difference in RMSEs.

    Returns
    -------
    None
        Only plots a graph.
    """

    # Aggregate the train, val, and test RMSEs.
    rmse_df = getperformance.get_overall_train_val_test_errors(
        horizons=horizons)

    # Prepare figure and axis.
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title(
        f'%-Difference in RMSEs Between Best Indiv. and Ensemble Model '
            'Per Forecast Horizon',
        fontsize=20)
    ax.set_xlabel('Forecast horizon (minutes)', fontsize=16)
    ax.set_ylabel('%-Difference in RMSE', fontsize=16)
    ax.tick_params(labelsize=14)
    colors = cm.tab10.colors
    markers = ['o', 'v', '^']

    # Prepare list of datasets, indiv models, and ensemble models.
    datasets = ['Train', 'Val', 'Test']
    indivs = ['images', 'images3d', 'irradiance', 'weather', 'combined', 'persistence']
    ensembles = ['linear ensemble', 'NN ensemble']

    # Initialize the maximum %-diff in RMSE needed for text placement.
    max_diff = 0

    # Iterate through each dataset and horizon and each dataset.
    for i, dataset in enumerate(datasets):

        rmses_diff = [] #we'll store the %-diff here

        for horizon in horizons:

            # Identify the best indiv. and ensemble model.
            # For this, we use the best validation RMSE.
            best_indiv = rmse_df.loc[indivs, (f'y_{horizon}', 'Val')].idxmin()
            best_ensemble = rmse_df.loc[ensembles, (f'y_{horizon}', 'Val')].idxmin()

            # Get the RMSE of the specific dataset for the best models.
            rmse_indiv = rmse_df.loc[best_indiv, (f'y_{horizon}', dataset)]
            rmse_ensemble = rmse_df.loc[best_ensemble, (f'y_{horizon}', dataset)]

            # Calculate the %-difference and add it to our list.
            perc_diff = ((rmse_ensemble - rmse_indiv) / rmse_indiv) * 100
            rmses_diff.append(perc_diff)

            # Change the maximum difference if needed.
            if perc_diff < max_diff:
                max_diff = perc_diff

        # Plot the %-differences.
        ax.plot(
            [str(j) for j in horizons], #makes plot more readable
            rmses_diff, #y-values
            marker=markers[i],
            color=colors[i],
            label=dataset)
        
    # We also want to plot vertical lines to indicate where the 
    # different forecast horizon buckets start. For this, we 
    # need the position of the buckets in the specified horizons list.
    intra_hour = [i<60 for i in horizons]
    intra_day = [i<60*24 for i in horizons]

    # We need to find the last horizon within each bucket. For this,
    # reversing the lists can help.
    intra_hour = intra_hour[::-1]
    intra_day = intra_day[::-1]

    # Now we can find the last instance within each bucket
    last_intra_hour = len(intra_hour) - 1 - intra_hour.index(True)
    last_intra_day = len(intra_day) - 1 - intra_day.index(True)

    # Plot vertical lines to indicate the forecast horizon buckets.
    ax.axvline(
        x=last_intra_hour+0.5, #line between intra-hour and intra-day
        linestyle='--',
        dashes=(5, 10),
        color='dimgrey')
    ax.axvline(
        x=last_intra_day+0.5, #line between intra-day and day-ahead
        linestyle='--',
        dashes=(5, 10),
        color='dimgrey')

    # Add labels to the lines.
    ax.text(
        x=last_intra_hour+0.45,
        y=max_diff,
        s='<- intra-hour',
        color='black',
        horizontalalignment='right',
        fontsize='large')
    ax.text(
        x=last_intra_day+0.55,
        y=max_diff,
        s='day-ahead ->',
        color='black',
        horizontalalignment='left',
        fontsize='large')
    ax.tick_params(labelsize=14)

    # Final enhancement to figure.
    plt.legend(loc='best', fontsize=16)
    fig.tight_layout()

