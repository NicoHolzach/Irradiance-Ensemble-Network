"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
main.py uses all the relevant functions
from the modules data.py, getperformance.py, 
models.py, plotting.py, and preprocessing.py.

With them, it conducts the analysis on the same
data and in, roughly, the same chronological order 
as the analysis conducted in the paper:

..................................................
CNN, LSTM, and Ensemble Neural Network 
Solar Irradiance Forecasting: Assessing the Impact 
of Model Architecture, Input Type, 
and Forecast Horizon
..................................................

The goal of this file and of the paper is to
train individual solar irradiance forecast 
CNN and LSTM models to predict
GHI values on a large set of forecast horizon.
These models are then combined into one
stacked ensemble model.

The following list maps the the individual sections 
in this main.py file to the individual sections in
the paper (if they can be mapped):


main.py section     :       paper section

1.2 - 1.5           :       4.2.2
2A                  :       4.2.3 i)
2B                  :       5.1.1
3A                  :       4.2.3 ii)
3B                  :       5.1.2
4A                  :       4.2.3 iii)
4B                  :       5.1.3
5A                  :       4.2.3 iv)      
5B                  :       5.1.4
6.1 - 6.2           :       5.1.5 i)
6.3 - 6.5           :       5.1.5 ii)
7                   :       5.2.1
8.1                 :       4.3.3 i)
8.2 - 8.4           :       5.2.2 i)
9.1                 :       4.3.3 ii)
9.2 - 9.5           :       5.2.2 ii)
10.1                :       4.3.3 iii)
10.2 - 10.3         :       5.2.2 iii)
11.1                :       4.3.3 iv)
11.2                :       5.2.2 iv)
12.1                :       5.3.1
12.2 - 12.3         :       5.3.2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# Import our own modules.
import data
import preprocessing
import models
import plotting
import getperformance

# Import other libraries.
import os
import time
import gc
import torch
import importlib
import itertools
importlib.reload(data) #use in VSCode to update modules)

# Ensure that CUDNN is enabled.
torch.backends.cudnn.benchmark=True

print('hello world :)')



# SET IMPORTANT GLOBAL VARIABLES
# THESE VARIABLES WILL HEAVILY IMPACT THE SPEED OF THIS FILE
GPU_ID = 2                  #which GPU to train on (0, 1, or 2).
                            #decide which GPU with this in terminal:
                            #watch -n 1 nvidia-smi
                            #note that kernel must be restarted to change the GPU.

DATA_FROM_SCRATCH = False   #whether to aggregate data from scratch.
                            #otherwise, use existing .npy and .pkl files.
                            #recommended to keep at False.

TRAIN_MODELS = True        #whether to train the models from scratch.
                            #can be set to True as the functins should
                            #see the already trained parameters in the
                            #./parameters folder.

PLOT_GRAPHS = False         #whether to plot all graphs or not.
                            #recommended to keep at False
                            #as aggregating the data for each graph
                            #can take some time.
                            #the graphs can also be found in the thesis.

# Change environment based on these variables.
# Do NOT modify this line.
os.environ['CUDA_VISIBLE_DEVICES'] = f'{GPU_ID}'
print(f'Selected GPU #{GPU_ID} for training.')


"""
################################################################################
#############
############# (1) CREATE AND EXPAND DATASETS
#############
################################################################################
"""

################################################################################
############# (1.1) CREATE DATASETS (if needed)
################################################################################

# Create datasets of skyimages, weather, and irradiance data from scratch.
if DATA_FROM_SCRATCH == True:

    print('Creating skyimage, irradiance, and weather datasets from scratch.')
    print('This might take a while...')

    # Create individual DataSet class objects.
    images = data.create_dataset(type='images',
                                from_pkl=False)
    irradiance = data.create_dataset(type='irradiance',
                                from_pkl=False)
    weather = data.create_dataset(type='weather',
                                from_pkl=False)

    # Save each dataset as a pickle file.
    images.save_data(full=True)
    irradiance.save_data(full=True)
    weather.save_data(full=True)

    # Also save just the skyimages as an individual numpy array.
    images.save_data(full=False,
                        datatype='images',
                        return_shape=False)


################################################################################
############# (1.2) CREATE FORECAST HORIZONS
################################################################################

# First, let's create the target values for our models.
# For this, create list of the following forecasting horizons:
# 15min, 30min, 45min, 1h, 2h, 3h, 6h, 12h, 1d, 2d, 3d
all_horizons = [15, 30, 45, 60, 2*60, 3*60, 6*60, 12*60, 24*60, 2*24*60, 3*24*60]

# Create irradiance variable if needed.
if 'irradiance' not in globals():
    irradiance = data.create_dataset(type='irradiance',
                                    from_pkl=True)

# Iterate through horizons and create targets for each of them.
print('Starting to create y targets for different forecast horizons...')
for horizon in all_horizons:
    
    # Start time counter and create targets.
    start_time = time.time()
    num_targets = irradiance.create_targets(horizon)

    # Save runtime for each dataset and forecast horizon.
    end_time = time.time()
    if num_targets is not None:
        runtime = end_time - start_time
        print(f'...Finished creating target y_{horizon}.')
        with open('./runtime/create_targets_runtime.txt', 'a') as f:
            f.write(f'y_{horizon};{runtime};{num_targets}\n')
    
    # Clear memory.
    gc.collect()
    print('-----------------------------')


################################################################################
############# (1.3) EXPAND IRRADIANCE DATA
################################################################################

# Second, expand the historical irradiance sequences for the LSTM.
# For this, create a list for all deltas, i.e. time (in minutes) between data points.
# Also create a list of all lengths, i.e. the length of the sequences that we want to consider.
all_deltas = [1, 3, 5, 10, 15]
all_lengths = [10, 25, 50, 100]

# Create irradiance variable if needed.
if 'irradiance' not in globals():
    irradiance = data.create_dataset(type='irradiance',
                                    from_pkl=True)

# For all delta-length combinations, create the respective irradiance sequence.
print('Starting to create irradiance sequences for different deltas and lengths...')
for delta, length in itertools.product(all_deltas, all_lengths):

    # Start time counter and create respective irradiance sequence.
    start_time = time.time()
    seq_shape = irradiance.create_irradiance(delta=delta, length=length)

    # Save runtime for each delta and length combination.
    end_time = time.time()
    if seq_shape is not None:
        runtime = end_time - start_time
        print(f'...Finished creating irradiance sequence with delta: {delta}min and length: {length}.')
        with open('./runtime/create_irradiance_runtime.txt', 'a') as f:
            f.write(f'irradiance_{delta}_{length};{runtime};{seq_shape}\n')

    # Clear memory.
    gc.collect()
    print('-----------------------------')


################################################################################
############# (1.4) EXPAND WEATHER DATA
################################################################################

# Third, expand the historical weather sequences for the LSTM.
# We will use the same deltas and lengths as for the irradiance sequences.
# Again, create weather variable if needed.
if 'weather' not in globals():
    weather = data.create_dataset(type='weather',
                                    from_pkl=True)

# For all delta-length combinations, create the respective weather sequence.
print('Starting to create weather sequences for different deltas and lengths...')
for delta, length in itertools.product(all_deltas, all_lengths):

    # Start time counter and create respective weather sequence.
    start_time = time.time()
    seq_shape = weather.create_weather(delta=delta, length=length)

    # Save runtime for each delta and length combination.
    end_time = time.time()
    if seq_shape is not None:
        runtime = end_time - start_time
        print(f'...Finished creating weather sequence with delta: {delta}min and length: {length}.')
        with open('./runtime/create_weather_runtime.txt', 'a') as f:
            f.write(f'weather_{delta}_{length};{runtime};{seq_shape}\n')

    # Clear memory.
    gc.collect()
    print('-----------------------------')


################################################################################
############# (1.5) EXPAND IMAGE DATA
################################################################################

# Lastly, expand the historical skyimage sequences for the 3D CNN.
# Let's define a list of delta and length of the 3d input we want to train with.
deltas_3d = [10, 30, 60]
lengths_3d = [3]

# Create images variable if needed.
if 'images' not in globals():
    images = data.create_dataset(type='images',
                                    from_pkl=True)

# For all delta-length combinations, create the respective skyimage sequence.
# Note: this can take some time.
print('Starting to create skyimage sequences for different deltas and lengths...')
for delta, length in itertools.product(deltas_3d, lengths_3d):

    # Start time counter and create respective skyimage sequence.
    start_time = time.time()
    seq_shape = images.create_images3d(delta=delta, length=length)

    # Save runtime for each delta and length combination.
    end_time = time.time()
    if seq_shape is not None:
        runtime = end_time - start_time
        print(f'...Finished creating skyimage sequence with delta: {delta}min and length: {length}.')
        with open('./runtime/create_images3d_runtime.txt', 'a') as f:
            f.write(f'images3d_{delta}_{length};{runtime};{seq_shape}\n')

    # Clear memory.
    gc.collect()
    print('-----------------------------')


################################################################################
############# (1.6) GET SUMMARY STATISTICS
################################################################################

# Print aggregated summary statistics for all data types and the
# three years of the train, val, test sets.
summary_df = data.summarize_data()

print('These are the summary statistics for each variable and each year: \n')
print(summary_df)


"""
################################################################################
#############
############# (2A) TRAIN DIFFERENT MODEL LAYER ARCHITECTURES
#############
################################################################################
"""

################################################################################
############# (2A.1) TRAIN CNN ARCHITECTURES W/ DATATYPE = IMAGES
################################################################################

# In this first training step, we only care about training different model architectures
# hence, we keep the hyperparameters the same for models (except the 3DCNN).
hyperparams = {'batch_size':[128],
                'lr':[1e-04],
                'weight_decay':[1e-05]}

# Define all possible parameters for the CNN layer architecture
# that we want to try out in training:
all_cnn_num = [2, 3, 4] #test 2, 3, and 4 convolutional layers
all_conv_filters = [4, 8, 16, 32, 64] #test these filter sizes
all_conv_kernels = [2, 4, 6] #these kernels for the filters
all_conv_strides = [1, 3, 5] #these strides for the filters
conv_padding = 1 #same paddding across models to decrease number of models to train
maxpool_kernel = 2 #again, only use one value to reduce number of models
maxpool_stride = 1
maxpool_padding = 1
all_dense_num = [1, 2, 3] #let's test three different dense layers at the end of the CNN
all_dense_dropout = [0.3, 0.5, 0.8] #and also three different dropout values
dense_activation = torch.nn.ReLU() #only one activation function to reduce number of models
dense_hidden_size = 100 #again, try to reduce number of models

# Use all these possible values to create a list of dictionaries
# where each dictionary contains a different set of these parameters
# that are used to initialize the model architecture.
cnn_layers_list = models.create_CNN_permutations(
    all_cnn_num=all_cnn_num, 
    all_conv_filters=all_conv_filters, 
    all_conv_kernels=all_conv_kernels, 
    all_conv_strides=all_conv_strides, 
    conv_padding=conv_padding, 
    maxpool_kernel=maxpool_kernel, 
    maxpool_stride=maxpool_stride, 
    maxpool_padding=maxpool_padding, 
    all_dense_num=all_dense_num, 
    all_dense_dropout=all_dense_dropout, 
    dense_activation=dense_activation, 
    dense_hidden_size=dense_hidden_size)

# Train each of these CNN models using skyimages.
if TRAIN_MODELS == True:

    models.train_multiple_models(
        modeltype='CNN',
        datatype='images',
        sample=True, #otherwise GPU runs out of memory
        sample_size=0.02,
        GPU=True,
        n_epochs=50,
        layers_list=cnn_layers_list,
        delta_list=[0], #model takes in a single image as input
        length_list=[1], #model takes in a single image as input
        horizon_list=[60], #train all architectures on a 60-min-ahead forecast
        hyperparameter_dict=hyperparams,
        loss_fn=models.RMSELoss(),
        layers_only=True, #this speeds up the data loading
        early_stopping=False,
        from_scratch=True)

################################################################################
############# (2A.2) TRAIN 3D CNN ARCHITECTURES W/ DATATYPE = IMAGES3D
################################################################################

# We use a batch size of 128 to speed up the training in this step.
# However, for some 3D CNN models, this batch size will overflow the GPU.
# Thus, we need to use a batch size of 64 for the CNN with images3d data.
# All other hyperparameters will remain the same.
hyperparams_3d = {'batch_size':[64],
                'lr':[1e-04],
                'weight_decay':[1e-05]}

# We will use the same layer architectures as with the 2D model.
# Train the 3D CNN model using skyimage sequences.
if TRAIN_MODELS == True:
        
    models.train_multiple_models(
        modeltype='CNN',
        datatype='images3d',
        sample=True, #otherwise GPU runs out of memory
        sample_size=0.02,
        GPU=True,
        n_epochs=50,
        layers_list=cnn_layers_list,
        delta_list=[10], #keep delta fixed at 10mins
        length_list=[3], #keep length of sequence fixed at 3
        horizon_list=[60], #train all architectures on a 60-min-ahead forecast
        hyperparameter_dict=hyperparams_3d,
        loss_fn=models.RMSELoss(),
        layers_only=True, #this speeds up the data loading
        early_stopping=False,
        from_scratch=True)


################################################################################
############# (2A.3) TRAIN LSTM ARCHITECTURES W/ DATATYPE = IRRADIANCE
################################################################################

# Define all possible parameters for the LSTM layer architecture
# that we want to try out in training:
all_lstm_hidden_size = [64, 128, 256] #test 3 different hidden unit sizes
all_lstm_num_layers = [2, 3, 4]
all_lstm_dropout = [0.25, 0.5, 0.75]

# We will use the same values for the dense layer paramaters
# that we used for the CNNs.
all_dense_num = [1, 2, 3] 
all_dense_dropout = [0.3, 0.5, 0.8] 
dense_activation = torch.nn.ReLU() 
dense_hidden_size = 100 

# Create list of different LSTM architectures that should be trained.
lstm_layers_list = models.create_LSTM_permutations(
    all_lstm_hidden_size=all_lstm_hidden_size,
    all_lstm_num_layers=all_lstm_num_layers, 
    all_lstm_dropout=all_lstm_dropout, 
    all_dense_num=all_dense_num, 
    all_dense_dropout=all_dense_dropout, 
    dense_activation=dense_activation, 
    dense_hidden_size=dense_hidden_size)

# Train the LSTM model using irradiance sequences.
if TRAIN_MODELS == True:
    
    models.train_multiple_models(
        modeltype='LSTM',
        datatype='irradiance',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=50,
        layers_list=lstm_layers_list,
        delta_list=[3], #keep delta fixed at 3min
        length_list=[25], #keep length of sequence fixed at 25
        horizon_list=[60], #train all architectures on a 60-min-ahead forecast
        hyperparameter_dict=hyperparams,
        loss_fn=models.RMSELoss(),
        layers_only=True, #this speeds up the data loading
        early_stopping=False,
        from_scratch=True)


################################################################################
############# (2A.4) TRAIN LSTM ARCHITECTURES W/ DATATYPE = WEATHER
################################################################################

# We will use the same layer architectures as with the LSTM irradiance model.
# Train the LSTM model using weather sequences:

if TRAIN_MODELS == True:
    
    models.train_multiple_models(
        modeltype='LSTM',
        datatype='weather',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=50,
        layers_list=lstm_layers_list,
        delta_list=[3], #keep delta fixed at 3min
        length_list=[25], #keep length of sequence fixed at 25
        horizon_list=[60], #train all architectures on a 60-min-ahead forecast
        hyperparameter_dict=hyperparams,
        loss_fn=models.RMSELoss(),
        layers_only=True, #this speeds up the data loading
        early_stopping=False,
        from_scratch=True)


################################################################################
############# (2A.5) TRAIN LSTM ARCHITECTURES W/ DATATYPE = COMBINED
################################################################################

# We will use the same layer architectures as with the other LSTMs.
# Train the LSTM model using irradiance + weather sequences:

if TRAIN_MODELS == True:
    
    models.train_multiple_models(
        modeltype='LSTM',
        datatype='combined',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=50,
        layers_list=lstm_layers_list,
        delta_list=[3], #keep delta fixed at 3min
        length_list=[25], #keep length of sequence fixed at 25
        horizon_list=[60], #train all architectures on a 60-min-ahead forecast
        hyperparameter_dict=hyperparams,
        loss_fn=models.RMSELoss(),
        layers_only=True, #this speeds up the data loading
        early_stopping=False,
        from_scratch=True)


"""
################################################################################
#############
############# (2B) ANALYZE PERFORMANCE OF THESE MODEL ARCHITECTURES
#############
################################################################################
"""

# In this section, we will analyze different parts of the models
# we just trained to see which type of model performs best.
# For this, let's define several lists of the particular parameters
# that we used to train the different types of model 
# layer architectures.
all_datatypes = ['images', 'images3d', 'irradiance', 'weather', 'combined']
deltas_of_layers = [0, 10, 3, 3, 3] #the images used delta=0,
                                    #images3d used delta=10,
                                    #irradiance: delta=3, etc.
lengths_of_layers = [1, 3, 25, 25, 25] #lengths we used 
batch_sizes_of_layers = [128, 64, 128, 128, 128] #we had to use a minibatch
                                                #size of 64 for images3d
lrs_of_layers = [1e-04] * 5 #same learning rate for all
weight_decays_of_layers = [1e-05] * 5 #same weight decay for all



################################################################################
############# (2B.1) COMPARE VALIDATION ERROR PER DATATYPE
################################################################################

# Let's compare the best mini-batch validation RMSE, or rather the mean of the 5 minimum
# RMSEs across any epoch per datatype. We use this metric, rather
# than just looking at the mini-batch validation RMSE of the last epoch as we have not
# tuned the hyperaparameters, yet and have not implemented an early-stopping rule
# in this step of the training (all following ones will have such a rule implemented). 
# Hence, each model might not have converged/ might have heavy fluctuations in the
# mini-batch validation RMSE from epoch to epoch.
# We will compare the five individual models to the persistence model.

# Get the validation RMSE per datatype of each architecture that performed the best.
rmse_of_layers = getperformance.compare_best_rmses(
    datatypes=all_datatypes,
    horizons=60,
    batch_sizes=batch_sizes_of_layers,
    lrs=lrs_of_layers,
    weight_decays=weight_decays_of_layers,
    include_epochs=True,
    min_or_mean='mean', #use the mean of the 5 best RMSEs as we have not implemented early stopping here
    include_persistence=True) #compare against the persistence model

print('This is the best RMSE (in W/m^2) of all trained architectures '
        'for every datatatype trained on a 1-hour-ahead forecast: \n')
print(rmse_of_layers)


################################################################################
############# (2B.2) GET PERFORMANCE OVERVIEW OF ARCHITECTURES PER DATATYPE
################################################################################

# Let's print the min, mean, and std. of the mini-batch validation RMSEs
# for each of the five datatypes across all of their trained architectures.
# This will give us a feel of potential model uncertainty in this step of the
# whole training process.
indiv_perf_of_layers = getperformance.get_indiv_performance_of_layers()

print('This is the min., mean, and std. of the mini-batch validation RMSEs '
    'of all five datatypes across their trained layer architectures: \n')
print(indiv_perf_of_layers)


################################################################################
############# (2B.3) PLOT RMSE HISTOGRAM PER DATATYPE
################################################################################

# To get an even better feel for this variability, let's plot the histograms
# of the mini-batch validation RMSEs in this training step for each datatype.
if PLOT_GRAPHS == True:

    # Plot the RMSE histograms
    plotting.plot_rmse_hist(
        datatypes=all_datatypes,
        deltas=deltas_of_layers,
        lengths=lengths_of_layers,
        horizon=60,
        batch_sizes=batch_sizes_of_layers,
        lrs=lrs_of_layers,
        weight_decays=weight_decays_of_layers,
        min_or_mean='mean') #we did not use early stopping in this step
                            #hence we should aggregate the RMSEs by using
                            #the mean of the lowest RMSEs per model.


################################################################################
############# (2B.4) COMPARE RMSE ACROSS DIFFERENT LAYER PARTS
################################################################################

# Given this large potential in performance gain, let's check which
# parameter of the layer architectures has the biggest impact.
if PLOT_GRAPHS == True:
    
    # Plot the performance of the CNN layers with image data.
    plotting.plot_performance_by_layer_parts(
        layer_parts=[
            'cnn_num',
            'conv_filters',
            'conv_kernels',
            'conv_strides',
            'dense_num',
            'dense_dropout'],
        datatype='images', 
        delta=0, 
        length=1, 
        horizon=60,
        batch_size=128, 
        lr=1e-04, 
        weight_decay=1e-05)

    # Plot the performance of the CNN layers with 3d image data.
    plotting.plot_performance_by_layer_parts(
        layer_parts=[
            'cnn_num',
            'conv_filters',
            'conv_kernels',
            'conv_strides',
            'dense_num',
            'dense_dropout'],
        datatype='images3d', 
        delta=10, 
        length=3, 
        horizon=60,
        batch_size=64, #had to use minibatches of size 64 due to GPU constraints
        lr=1e-04, 
        weight_decay=1e-05)

    # Plot the performance of the LSTM layers with irradiance data.
    plotting.plot_performance_by_layer_parts(
        layer_parts=[
            'lstm_hidden_size',
            'lstm_num_layers',
            'lstm_dropout',
            'dense_num',
            'dense_dropout'],
        datatype='irradiance', 
        delta=3, 
        length=25, 
        horizon=60,
        batch_size=128, 
        lr=1e-04, 
        weight_decay=1e-05)

    # Plot the performance of the LSTM layers with weather data.
    plotting.plot_performance_by_layer_parts(
        layer_parts=[
            'lstm_hidden_size',
            'lstm_num_layers',
            'lstm_dropout',
            'dense_num',
            'dense_dropout'],
        datatype='weather', 
        delta=3, 
        length=25, 
        horizon=60,
        batch_size=128, 
        lr=1e-04, 
        weight_decay=1e-05)

    # Plot the performance of the LSTM layers with combined data.
    plotting.plot_performance_by_layer_parts(
        layer_parts=[
            'lstm_hidden_size',
            'lstm_num_layers',
            'lstm_dropout',
            'dense_num',
            'dense_dropout'],
        datatype='combined', 
        delta=3, 
        length=25, 
        horizon=60,
        batch_size=128, 
        lr=1e-04, 
        weight_decay=1e-05)


################################################################################
############# (2B.5) CHECK BEST PERFORMING LAYER ARCHITECTURE PER DATATYPE
################################################################################

# Let's check which layer architecture generated the best validation set 
# performance for each datatype.
best_layers_df = getperformance.aggregate_best_layers(
    datatypes=all_datatypes,
    deltas=deltas_of_layers,
    lengths=lengths_of_layers,
    horizon=60,
    batch_sizes=batch_sizes_of_layers,
    lrs=lrs_of_layers,
    weight_decays=weight_decays_of_layers)

print('This is the best performing layers architecture '
        'for every datatatype trained on an 1-hour-ahead forecast: \n')
print(best_layers_df)


################################################################################
############# (2B.6) PLOT THE TRUE Y VS. THE PREDICTED YHAT
################################################################################

# To get a first impression as to whether the predictions of each model
# are similarly correlated or not with the true targets, let's plot each prediction
# of each best model per datatype against the true targets.
if PLOT_GRAPHS == True:

    # Plot y_true vs yhat of the validation set.
    plotting.plot_ytrue_yhat(
        datatypes=all_datatypes,
        deltas=deltas_of_layers,
        lengths=lengths_of_layers,
        horizon=60,
        batch_sizes=batch_sizes_of_layers,
        lrs=lrs_of_layers,
        weight_decays=weight_decays_of_layers)


################################################################################
############# (2B.7) PLOT TRAIN AND VAL ERROR PER DATATYPE
################################################################################

# Let's plot the  train and validation RMSEs for each datatype per epoch 
# to check the convergence of the models during training.

if PLOT_GRAPHS == True:

    # Plot losses of the best layer architecture per datatype.
    plotting.plot_best_train_val(
        datatypes=all_datatypes,
        deltas=deltas_of_layers,
        lengths=lengths_of_layers,
        horizon=60,
        batch_sizes=batch_sizes_of_layers,
        lrs=lrs_of_layers,
        weight_decays=weight_decays_of_layers)


"""
################################################################################
#############
############# (3A) TRAIN DIFFERENT HORIZONS WITH VARYING INPUT
#############
################################################################################
"""

# We have now determined the best layer architecture for each
# datatype (i.e. images, images3d, irradiance, weather, combined).
# However, we only trained each model on one particular input
# i.e. one value for delta and length each. We also only trained
# them on y=60, i.e. a 1-hour-ahead forecast.
# In this step, we will train models on three forecast horizons:
# intra-hour, intra-day, and day-ahead and check which type of input 
# data is the most suitable for which length of forecast horizon.

# First, we need to get the layer parameters of each layer architecture
# that performed the best per datatype. 
print('Collecting the best layer architectures per datatype...')
best_layers_images = getperformance.get_best_layers(
    datatype='images',
    return_layer_params=True,
    delta=0,
    length=1,
    horizon=60,
    batch_size=128,
    lr=1e-04,
    weight_decay=1e-05)
best_layers_images3d = getperformance.get_best_layers(
    datatype='images3d',
    return_layer_params=True,
    delta=10, #we trained our 3d cnn using image sequences with delta 10
    length=3, #these sequences were of length 3
    horizon=60, #we trained the model on a 1-hour-ahead forecast
    batch_size=64, #due to GPU constraints, we used minibatches of size 64
    lr=1e-04,
    weight_decay=1e-05)
best_layers_irradiance = getperformance.get_best_layers(
    datatype='irradiance',
    return_layer_params=True,
    delta=3, #we used the same delta for all LSTM models (irradiance, weather, combined)
    length=25, #same with the length
    horizon=60,
    batch_size=128,
    lr=1e-04,
    weight_decay=1e-05)
best_layers_weather = getperformance.get_best_layers(
    datatype='weather',
    return_layer_params=True,
    delta=3, 
    length=25, 
    horizon=60,
    batch_size=128,
    lr=1e-04,
    weight_decay=1e-05)
best_layers_combined = getperformance.get_best_layers(
    datatype='combined',
    return_layer_params=True,
    delta=3, 
    length=25, 
    horizon=60,
    batch_size=128,
    lr=1e-04,
    weight_decay=1e-05)

# We now need to define a list of forecast horizons that we want
# to use as our targets for training. The literature seems to 
# divide the forecast horizon into intra-hour, intra-day, and 
# day-ahead forecasts. Thus, using a 30-min-, 2-hour-, and 1-day-
# ahead target seem like reasonable and interesting horizons for
# each of these buckets.
horizon_buckets = [30, 120, 60*24] #30mins, 2hours, 1day

# We will test all different deltas and lengths that we have
# created above for the data that is fed into the LSTM.
all_deltas = [1, 3, 5, 10, 15]
all_lengths = [10, 25, 50, 100]

# Similarly for the CNN that gets fed images3d data,
# we will test all different deltas of images3d sequences
# we have created above. Note that this is not relevant
# for the CNN with images data as, by design, it only gets fed
# a single image and therefore always has delta=0, length=1.
deltas_3d = [10, 30, 60]

# To improve convergence of the longer sequences, we will
# increase the learning rate. 
# All other hyperparamters will stay the same.
hyperparams_horizons_lstm = {'batch_size':[128],
                            'lr':[3e-04], #higher lr here compared to before
                            'weight_decay':[1e-05]}

hyperparams_horizons_cnn = {'batch_size':[128],
                            'lr':[1e-04], #lr can stay low as the CNNs seem to converge quickly
                            'weight_decay':[1e-05]}

hyperparams_horizons_3d = {'batch_size':[64], #again, to reduce load on GPU
                            'lr':[1e-04],
                            'weight_decay':[1e-05]}


################################################################################
############# (3A.1) TRAIN CNN (IMAGES) FOR VARYING HORIZONS
################################################################################

# Note that this type of model only gets fed a single image 
# and no image sequence. Hence, we can only vary the forecast
# horizon and not the input.

# Train a CNN with images data for each horizon:
if TRAIN_MODELS == True:

    models.train_multiple_models(
        modeltype='CNN',
        datatype='images',
        sample=True, 
        sample_size=0.02,
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_images], #the parameters of the best architecture
        delta_list=[0], 
        length_list=[1], 
        horizon_list=horizon_buckets, #train one model for each horizon bucket
        hyperparameter_dict=hyperparams_horizons_cnn,
        loss_fn=models.RMSELoss(),
        layers_only=False, #as the data now varies, we cannot use this anymore
        early_stopping=True, #will stop training at convergence
        from_scratch=False) 


################################################################################
############# (3A.2) TRAIN 3D CNN (IMAGES3D) FOR VARYING INPUT AND HORIZONS
################################################################################

# Train a 3D CNN with images3d data for each delta and horizon:
if TRAIN_MODELS == True:
        
    models.train_multiple_models(
        modeltype='CNN',
        datatype='images3d',
        sample=True, #otherwise GPU runs out of memory
        sample_size=0.02,
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_images3d], #the parameters of the best architecture
        delta_list=deltas_3d, #varying deltas
        length_list=[3], #we use a fixed length here to keep memory lower
        horizon_list=horizon_buckets, #train one model for each horizon bucket
        hyperparameter_dict=hyperparams_horizons_3d,
        loss_fn=models.RMSELoss(),
        layers_only=False, #as the data now varies, we cannot use this anymore
        early_stopping=True,
        from_scratch=False) 


################################################################################
############# (3A.3) TRAIN LSTM (IRRADIANCE) FOR VARYING INPUT AND HORIZONS
################################################################################

# Train an LSTM with irradiance data for each delta, length, and horizon:
if TRAIN_MODELS == True:
    
    models.train_multiple_models(
        modeltype='LSTM',
        datatype='irradiance',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_irradiance], #the parameters of the best architecture
        delta_list=all_deltas, #varying deltas
        length_list=all_lengths, #varying lengths
        horizon_list=horizon_buckets, #train one model for each horizon bucket
        hyperparameter_dict=hyperparams_horizons_lstm,
        loss_fn=models.RMSELoss(),
        layers_only=False, #as the data now varies, we cannot use this anymore
        early_stopping=True,
        from_scratch=False)


################################################################################
############# (3A.4) TRAIN LSTM (WEATHER) FOR VARYING INPUT AND HORIZONS
################################################################################

# Train an LSTM with weather data for each delta, length, and horizon:
if TRAIN_MODELS == True:
    
    models.train_multiple_models(
        modeltype='LSTM',
        datatype='weather',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500,
        layers_list=[best_layers_weather], #the parameters of the best architecture
        delta_list=all_deltas, #varying deltas
        length_list=all_lengths, #varying lengths
        horizon_list=horizon_buckets, #train one model for each horizon bucket
        hyperparameter_dict=hyperparams_horizons_lstm,
        loss_fn=models.RMSELoss(),
        layers_only=False, #as the data now varies, we cannot use this anymore
        early_stopping=True,
        from_scratch=False)


################################################################################
############# (3A.5) TRAIN LSTM (COMBINED) FOR VARYING INPUT AND HORIZONS
################################################################################

# Train an LSTM with irradiance+weather data for each delta, length, and horizon:
if TRAIN_MODELS == True:
    
    models.train_multiple_models(
        modeltype='LSTM',
        datatype='combined',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500,
        layers_list=[best_layers_combined], #the parameters of the best architecture
        delta_list=all_deltas, #varying deltas
        length_list=all_lengths, #varying lengths
        horizon_list=horizon_buckets, #train one model for each horizon bucket
        hyperparameter_dict=hyperparams_horizons_lstm,
        loss_fn=models.RMSELoss(),
        layers_only=False, #as the data now varies, we cannot use this anymore
        early_stopping=True,
        from_scratch=False)


"""
################################################################################
############# 
############# (3B) ANALYZING PERFORMANCE BY DELTA/LENGTH PER HORIZON
############# 
################################################################################
"""

# Again, these are the datatypes we trained the models with
# and the forecast horizons we trained the models on:
all_datatypes = ['images', 'images3d', 'irradiance', 'weather', 'combined']
horizon_buckets = [30, 120, 60*24] #30mins, 2hours, 1day

# And these are the hyperparameters we used to train the
# models with different deltas/lengths on these horizons.
batch_sizes_of_horizons = [128, 64, 128, 128, 128] #nothing changed here to before
lrs_of_horizons = [1e-04, 1e-04, 3e-04, 3e-04, 3e-04] #lr changed for lstm for faster convergence
weight_decays_of_horizons = [1e-05] * 5 #nothing changed here


################################################################################
############# (3B.1) GET PERFORMANCE OVERVIEW OF DELTA-LENGTHS PER DATATYPE
################################################################################

# Let's print the min, mean, and std. of the mini-batch validation RMSEs
# for each of the five datatypes across all of their trained delta-length inputs.
# This will give us a feel of potential model uncertainty in this step of the
# whole training process.

# Let's check for y_30.
indiv_perf_of_inputs_y30 = getperformance.get_indiv_performance_of_inputs(
    horizon=30)
print('This is the min., mean, and std. of the mini-batch validation RMSEs '
    'of all five datatypes across their trained delta-length inputs '
    'for the forecast horizon 30min: \n')
print(indiv_perf_of_inputs_y30)

# Let's check for y_120.
indiv_perf_of_inputs_y120 = getperformance.get_indiv_performance_of_inputs(
    horizon=120)
print('This is the min., mean, and std. of the mini-batch validation RMSEs '
    'of all five datatypes across their trained delta-length inputs '
    'for the forecast horizon 120min: \n')
print(indiv_perf_of_inputs_y120)

# Let's check for y_1440.
indiv_perf_of_inputs_y1440 = getperformance.get_indiv_performance_of_inputs(
    horizon=1440)
print('This is the min., mean, and std. of the mini-batch validation RMSEs '
    'of all five datatypes across their trained delta-length inputs '
    'for the forecast horizon 1440min: \n')
print(indiv_perf_of_inputs_y1440)


################################################################################
############# (3B.2) COMPARE VALIDATION ERROR PER DATATYPE AND HORIZON
################################################################################

# Let's again compare the best RMSEs across datatypes.
# Let's see which delta/length combination performed the best
# for which forecast horizon.
rmse_of_horizons = getperformance.compare_best_rmses(
    datatypes=all_datatypes,
    horizons=horizon_buckets,
    batch_sizes=batch_sizes_of_horizons,
    lrs=lrs_of_horizons,
    weight_decays=weight_decays_of_horizons,
    min_or_mean='min', #use the minimum/best RMSE 
    include_persistence=True, #compare against the persistence model
    include_epochs=True, #to show how long each model took until covergence
    include_delta_length=True) #show which input sequence performed the best

print('This is the best RMSE (in W/m^2) and the best performing (delta, length) '
        'combination for every datatatype trained on 3 different forecast horizons: \n')
print(rmse_of_horizons)


################################################################################
############# (3B.3) ANALYZE RELATIONSHIP BETWEEN DELTA/LENGTH AND HORIZON
################################################################################

# Plot the RSMEs of the LSTM models by delta, length
# and compare them across horizons.
if PLOT_GRAPHS == True:
        
    plotting.plot_lstm_error_by_delta_length(
        datatypes=['irradiance', 'weather', 'combined'], 
        horizons=horizon_buckets,
        batch_sizes=[128]*3, #same hyperparams for all 3 LSTMs
        lrs=[3e-04]*3, 
        weight_decays=[1e-05]*3,
        min_or_mean='mean') #plot the mean RMSE per delta/length

    plotting.plot_lstm_error_by_delta_length(
        datatypes=['irradiance', 'weather', 'combined'], 
        horizons=horizon_buckets,
        batch_sizes=[128]*3, #same hyperparams for all 3 LSTMs
        lrs=[3e-04]*3, 
        weight_decays=[1e-05]*3,
        min_or_mean='min') #plot the best RMSE per delta/length

# Plot the impact of delta on the RSME for the 3D CNN.
if PLOT_GRAPHS == True:
        
    plotting.plot_cnn_error_by_delta(
        horizons=horizon_buckets,
        batch_size=64, 
        lr=1e-04, 
        weight_decay=1e-05)


"""
################################################################################
############# 
############# (4A) TUNING THE HYPERPARAMETERS 
############# 
################################################################################
"""

# First, we need to get the delta-length combination per horizon
# per datatype that has the best performance. Then we can use 
# this delta-length-horizon combination to tune the hyperparameters.

# Get the deltas and lengths for the horizon y_30.
_, delta_y30_img, length_y30_img = getperformance.get_best_model_by_horizon(
    datatype='images', horizon=30, return_delta_length=True, return_hyperparams=False,
    batch_size=128, lr=1e-04, weight_decay=1e-05)

_, delta_y30_img3d, length_y30_img3d = getperformance.get_best_model_by_horizon(
    datatype='images3d', horizon=30, return_delta_length=True, return_hyperparams=False,
    batch_size=64, lr=1e-04, weight_decay=1e-05) #we used a higher batch_size for the 3D CNN

_, delta_y30_irr, length_y30_irr = getperformance.get_best_model_by_horizon(
    datatype='irradiance', horizon=30, return_delta_length=True, return_hyperparams=False,
    batch_size=128, lr=3e-04, weight_decay=1e-05) #used a higher lr for the LSTMs

_, delta_y30_weath, length_y30_weath = getperformance.get_best_model_by_horizon(
    datatype='weather', horizon=30, return_delta_length=True, return_hyperparams=False,
    batch_size=128, lr=3e-04, weight_decay=1e-05) #used a higher lr for the LSTMs

_, delta_y30_comb, length_y30_comb = getperformance.get_best_model_by_horizon(
    datatype='combined', horizon=30, return_delta_length=True, return_hyperparams=False,
    batch_size=128, lr=3e-04, weight_decay=1e-05) #used a higher lr for the LSTMs

# Get the deltas and lengths for the horizon y_120.
_, delta_y120_img, length_y120_img = getperformance.get_best_model_by_horizon(
    datatype='images', horizon=120, return_delta_length=True, return_hyperparams=False,
    batch_size=128, lr=1e-04, weight_decay=1e-05)

_, delta_y120_img3d, length_y120_img3d = getperformance.get_best_model_by_horizon(
    datatype='images3d', horizon=120, return_delta_length=True, return_hyperparams=False,
    batch_size=64, lr=1e-04, weight_decay=1e-05) #we used a higher batch_size for the 3D CNN

_, delta_y120_irr, length_y120_irr = getperformance.get_best_model_by_horizon(
    datatype='irradiance', horizon=120, return_delta_length=True, return_hyperparams=False,
    batch_size=128, lr=3e-04, weight_decay=1e-05) #used a higher lr for the LSTMs

_, delta_y120_weath, length_y120_weath = getperformance.get_best_model_by_horizon(
    datatype='weather', horizon=120, return_delta_length=True, return_hyperparams=False,
    batch_size=128, lr=3e-04, weight_decay=1e-05) #used a higher lr for the LSTMs

_, delta_y120_comb, length_y120_comb = getperformance.get_best_model_by_horizon(
    datatype='combined', horizon=120, return_delta_length=True, return_hyperparams=False,
    batch_size=128, lr=3e-04, weight_decay=1e-05) #used a higher lr for the LSTMs

# Get the deltas and lengths for the horizon y_1440.
_, delta_y1440_img, length_y1440_img = getperformance.get_best_model_by_horizon(
    datatype='images', horizon=1440, return_delta_length=True, return_hyperparams=False,
    batch_size=128, lr=1e-04, weight_decay=1e-05)

_, delta_y1440_img3d, length_y1440_img3d = getperformance.get_best_model_by_horizon(
    datatype='images3d', horizon=1440, return_delta_length=True, return_hyperparams=False,
    batch_size=64, lr=1e-04, weight_decay=1e-05) #we used a higher batch_size for the 3D CNN

_, delta_y1440_irr, length_y1440_irr = getperformance.get_best_model_by_horizon(
    datatype='irradiance', horizon=1440, return_delta_length=True, return_hyperparams=False,
    batch_size=128, lr=3e-04, weight_decay=1e-05) #used a higher lr for the LSTMs

_, delta_y1440_weath, length_y1440_weath = getperformance.get_best_model_by_horizon(
    datatype='weather', horizon=1440, return_delta_length=True, return_hyperparams=False,
    batch_size=128, lr=3e-04, weight_decay=1e-05) #used a higher lr for the LSTMs

_, delta_y1440_comb, length_y1440_comb = getperformance.get_best_model_by_horizon(
    datatype='combined', horizon=1440, return_delta_length=True, return_hyperparams=False,
    batch_size=128, lr=3e-04, weight_decay=1e-05) #used a higher lr for the LSTMs

# We will perform a grid-search over all these hyperparameters
# per delta-length-horizon combination.
all_hyperparams = {'batch_size':[32, 64, 128],
                    'lr':[1e-05, 5e-05, 1e-04, 5e-04, 1e-03],
                    'weight_decay':[1e-05, 1e-04, 1e-03]}


################################################################################
############# (4A.1) HYPERPARAMETER-TUNING OF CNN (IMAGES) FOR ALL 3 HORIZONS
################################################################################

# Tune the hyperparameters of the CNN with images data 
# for the horizon y_30:
if TRAIN_MODELS == True:

    models.train_multiple_models(
        modeltype='CNN',
        datatype='images',
        sample=True, 
        sample_size=0.02,
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_images], #the parameters of the best architecture
        delta_list=[delta_y30_img], #best performing delta for this horizon
        length_list=[length_y30_img], #best performing length for this horizon
        horizon_list=[30], #tune the hyperparams for this horizon
        hyperparameter_dict=all_hyperparams, #gridsearch over these
        loss_fn=models.RMSELoss(),
        layers_only=False, #as the data now varies, we cannot use this anymore
        early_stopping=True, #will stop training at convergence
        from_scratch=False) 

# Tune the hyperparameters of the CNN with images data 
# for the horizon y_120:
if TRAIN_MODELS == True:

    models.train_multiple_models(
        modeltype='CNN',
        datatype='images',
        sample=True, 
        sample_size=0.02,
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_images], #the parameters of the best architecture
        delta_list=[delta_y120_img], #best performing delta for this horizon
        length_list=[length_y120_img], #best performing length for this horizon
        horizon_list=[120], #tune the hyperparams for this horizon
        hyperparameter_dict=all_hyperparams, #gridsearch over these
        loss_fn=models.RMSELoss(),
        layers_only=False, #as the data now varies, we cannot use this anymore
        early_stopping=True, #will stop training at convergence
        from_scratch=False) 

# Tune the hyperparameters of the CNN with images data 
# for the horizon y_1440:
if TRAIN_MODELS == True:

    models.train_multiple_models(
        modeltype='CNN',
        datatype='images',
        sample=True, 
        sample_size=0.02,
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_images], #the parameters of the best architecture
        delta_list=[delta_y1440_img], #best performing delta for this horizon
        length_list=[length_y1440_img], #best performing length for this horizon
        horizon_list=[1440], #tune the hyperparams for this horizon
        hyperparameter_dict=all_hyperparams, #gridsearch over these
        loss_fn=models.RMSELoss(),
        layers_only=False, #as the data now varies, we cannot use this anymore
        early_stopping=True, #will stop training at convergence
        from_scratch=False) 


################################################################################
############# (4A.2) HYPERPARAMETER-TUNING OF 3D CNN (IMAGES3D) FOR ALL 3 HORIZONS
################################################################################

# Tune the hyperparameters of the CNN with images3d data 
# for the horizon y_30:
if TRAIN_MODELS == True:
        
    models.train_multiple_models(
        modeltype='CNN',
        datatype='images3d',
        sample=True, #otherwise GPU runs out of memory
        sample_size=0.02,
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_images3d], #the parameters of the best architecture
        delta_list=[delta_y30_img3d], #best performing delta for this horizon
        length_list=[length_y30_img3d], #best performing length for this horizon
        horizon_list=[30], #tune the hyperparams for this horizon
        hyperparameter_dict=all_hyperparams, #gridsearch over these
        loss_fn=models.RMSELoss(),
        layers_only=False, #as the data now varies, we cannot use this anymore
        early_stopping=True,
        from_scratch=False) 

# Tune the hyperparameters of the CNN with images3d data 
# for the horizon y_120:
if TRAIN_MODELS == True:
        
    models.train_multiple_models(
        modeltype='CNN',
        datatype='images3d',
        sample=True, #otherwise GPU runs out of memory
        sample_size=0.02,
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_images3d], #the parameters of the best architecture
        delta_list=[delta_y120_img3d], #best performing delta for this horizon
        length_list=[length_y120_img3d], #best performing length for this horizon
        horizon_list=[120], #tune the hyperparams for this horizon
        hyperparameter_dict=all_hyperparams, #gridsearch over these
        loss_fn=models.RMSELoss(),
        layers_only=False, #as the data now varies, we cannot use this anymore
        early_stopping=True,
        from_scratch=False) 

# Tune the hyperparameters of the CNN with images3d data 
# for the horizon y_1440:
if TRAIN_MODELS == True:
        
    models.train_multiple_models(
        modeltype='CNN',
        datatype='images3d',
        sample=True, #otherwise GPU runs out of memory
        sample_size=0.02,
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_images3d], #the parameters of the best architecture
        delta_list=[delta_y1440_img3d], #best performing delta for this horizon
        length_list=[length_y1440_img3d], #best performing length for this horizon
        horizon_list=[1440], #tune the hyperparams for this horizon
        hyperparameter_dict=all_hyperparams, #gridsearch over these
        loss_fn=models.RMSELoss(),
        layers_only=False, #as the data now varies, we cannot use this anymore
        early_stopping=True,
        from_scratch=False) 


################################################################################
############# (4A.3) HYPERPARAMETER-TUNING OF LSTM (IRRADIANCE) FOR ALL 3 HORIZONS
################################################################################

# Tune the hyperparameters of the LSTM with irradiance data 
# for the horizon y_30:
if TRAIN_MODELS == True:
    
    models.train_multiple_models(
        modeltype='LSTM',
        datatype='irradiance',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_irradiance], #the parameters of the best architecture
        delta_list=[delta_y30_irr], #best performing delta for this horizon
        length_list=[length_y30_irr], #best performing length for this horizon
        horizon_list=[30], #tune the hyperparams for this horizon
        hyperparameter_dict=all_hyperparams, #gridsearch over these
        loss_fn=models.RMSELoss(),
        layers_only=False, #as the data now varies, we cannot use this anymore
        early_stopping=True,
        from_scratch=False)

# Tune the hyperparameters of the LSTM with irradiance data 
# for the horizon y_120:
if TRAIN_MODELS == True:
    
    models.train_multiple_models(
        modeltype='LSTM',
        datatype='irradiance',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_irradiance], #the parameters of the best architecture
        delta_list=[delta_y120_irr], #best performing delta for this horizon
        length_list=[length_y120_irr], #best performing length for this horizon
        horizon_list=[120], #tune the hyperparams for this horizon
        hyperparameter_dict=all_hyperparams, #gridsearch over these
        loss_fn=models.RMSELoss(),
        layers_only=False, #as the data now varies, we cannot use this anymore
        early_stopping=True,
        from_scratch=False)

# Tune the hyperparameters of the LSTM with irradiance data 
# for the horizon y_1440:
if TRAIN_MODELS == True:
    
    models.train_multiple_models(
        modeltype='LSTM',
        datatype='irradiance',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_irradiance], #the parameters of the best architecture
        delta_list=[delta_y1440_irr], #best performing delta for this horizon
        length_list=[length_y1440_irr], #best performing length for this horizon
        horizon_list=[1440], #tune the hyperparams for this horizon
        hyperparameter_dict=all_hyperparams, #gridsearch over these
        loss_fn=models.RMSELoss(),
        layers_only=False, #as the data now varies, we cannot use this anymore
        early_stopping=True,
        from_scratch=False)


################################################################################
############# (4A.4) HYPERPARAMETER-TUNING OF LSTM (WEATHER) FOR ALL 3 HORIZONS
################################################################################

# Tune the hyperparameters of the LSTM with weather data 
# for the horizon y_30:
if TRAIN_MODELS == True:
    
    models.train_multiple_models(
        modeltype='LSTM',
        datatype='weather',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_weather], #the parameters of the best architecture
        delta_list=[delta_y30_weath], #best performing delta for this horizon
        length_list=[length_y30_weath], #best performing length for this horizon
        horizon_list=[30], #tune the hyperparams for this horizon
        hyperparameter_dict=all_hyperparams, #gridsearch over these
        loss_fn=models.RMSELoss(),
        layers_only=False, #as the data now varies, we cannot use this anymore
        early_stopping=True,
        from_scratch=False)

# Tune the hyperparameters of the LSTM with weather data 
# for the horizon y_120:
if TRAIN_MODELS == True:
    
    models.train_multiple_models(
        modeltype='LSTM',
        datatype='weather',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_weather], #the parameters of the best architecture
        delta_list=[delta_y120_weath], #best performing delta for this horizon
        length_list=[length_y120_weath], #best performing length for this horizon
        horizon_list=[120], #tune the hyperparams for this horizon
        hyperparameter_dict=all_hyperparams, #gridsearch over these
        loss_fn=models.RMSELoss(),
        layers_only=False, #as the data now varies, we cannot use this anymore
        early_stopping=True,
        from_scratch=False)

# Tune the hyperparameters of the LSTM with weather data 
# for the horizon y_1440:
if TRAIN_MODELS == True:
    
    models.train_multiple_models(
        modeltype='LSTM',
        datatype='weather',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_weather], #the parameters of the best architecture
        delta_list=[delta_y1440_weath], #best performing delta for this horizon
        length_list=[length_y1440_weath], #best performing length for this horizon
        horizon_list=[1440], #tune the hyperparams for this horizon
        hyperparameter_dict=all_hyperparams, #gridsearch over these
        loss_fn=models.RMSELoss(),
        layers_only=False, #as the data now varies, we cannot use this anymore
        early_stopping=True,
        from_scratch=False)


################################################################################
############# (4A.5) HYPERPARAMETER-TUNING OF LSTM (COMBINED) FOR ALL 3 HORIZONS
################################################################################

# Tune the hyperparameters of the LSTM with combined data 
# for the horizon y_30:
if TRAIN_MODELS == True:
    
    models.train_multiple_models(
        modeltype='LSTM',
        datatype='combined',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_combined], #the parameters of the best architecture
        delta_list=[delta_y30_comb], #best performing delta for this horizon
        length_list=[length_y30_comb], #best performing length for this horizon
        horizon_list=[30], #tune the hyperparams for this horizon
        hyperparameter_dict=all_hyperparams, #gridsearch over these
        loss_fn=models.RMSELoss(),
        layers_only=False, #as the data now varies, we cannot use this anymore
        early_stopping=True,
        from_scratch=False)

# Tune the hyperparameters of the LSTM with combined data 
# for the horizon y_120:
if TRAIN_MODELS == True:
    
    models.train_multiple_models(
        modeltype='LSTM',
        datatype='combined',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_combined], #the parameters of the best architecture
        delta_list=[delta_y120_comb], #best performing delta for this horizon
        length_list=[length_y120_comb], #best performing length for this horizon
        horizon_list=[120], #tune the hyperparams for this horizon
        hyperparameter_dict=all_hyperparams, #gridsearch over these
        loss_fn=models.RMSELoss(),
        layers_only=False, #as the data now varies, we cannot use this anymore
        early_stopping=True,
        from_scratch=False)

# Tune the hyperparameters of the LSTM with combined data 
# for the horizon y_1440:
if TRAIN_MODELS == True:
    
    models.train_multiple_models(
        modeltype='LSTM',
        datatype='combined',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_combined], #the parameters of the best architecture
        delta_list=[delta_y1440_comb], #best performing delta for this horizon
        length_list=[length_y1440_comb], #best performing length for this horizon
        horizon_list=[1440], #tune the hyperparams for this horizon
        hyperparameter_dict=all_hyperparams, #gridsearch over these
        loss_fn=models.RMSELoss(),
        layers_only=False, #as the data now varies, we cannot use this anymore
        early_stopping=True,
        from_scratch=False)


"""
################################################################################
############# 
############# (4B) ANALYZING THE PERFORMANCE BY HYPERPARAMETER
############# 
################################################################################
"""

################################################################################
############# (4B.1) GET PERFORMANCE OVERVIEW OF DELTA-LENGTHS PER DATATYPE
################################################################################

# Let's print the min, mean, and std. of the mini-batch validation RMSEs
# for each of the five datatypes across all of their trained hyperparameters.
# This will give us a feel of potential model uncertainty in this step of the
# whole training process.

# Let's check this for y_30.
indiv_perf_of_hyperparams_y30 = getperformance.get_indiv_performance_of_hyperparams(
    horizon=30)
print('This is the min., mean, and std. of the mini-batch validation RMSEs '
    'of all five datatypes across their trained hyperparameters '
    'with a forecast horizon of 30mins : \n')
print(indiv_perf_of_hyperparams_y30)

# Let's check this for y_120.
indiv_perf_of_hyperparams_y120 = getperformance.get_indiv_performance_of_hyperparams(
    horizon=120)
print('This is the min., mean, and std. of the mini-batch validation RMSEs '
    'of all five datatypes across their trained hyperparameters '
    'with a forecast horizon of 120mins : \n')
print(indiv_perf_of_hyperparams_y120)

# Let's check this for y_1440.
indiv_perf_of_hyperparams_y1440 = getperformance.get_indiv_performance_of_hyperparams(
    horizon=1440)
print('This is the min., mean, and std. of the mini-batch validation RMSEs '
    'of all five datatypes across their trained hyperparameters '
    'with a forecast horizon of 1440mins : \n')
print(indiv_perf_of_hyperparams_y1440)


################################################################################
############# (4B.2) PLOT HISTOGRAM OF THE RMSEs PER DATATYPE AND HORIZON BUCKET
################################################################################

# Let's further analyze this variablity by plotting some histograms
# for each datatype and horizon of this training step.
if PLOT_GRAPHS == True:

    plotting.plot_rmse_hist_by_horizon_of_hyperparams(
        datatypes=all_datatypes,
        horizons=horizon_buckets)

################################################################################
############# (4B.3) CHECK VALIDATION ERROR PER BEST HYPERPARAMETER
################################################################################

# Let's again compare the best RMSEs across datatypes.
# Let's see which delta/length combination performed the best
# for which forecast horizon.
rmse_of_hyperparams = getperformance.compare_best_rmses(
    datatypes=all_datatypes,
    horizons=horizon_buckets,
    batch_sizes=None, #check all batch sizes of each model
    lrs=None, #check all learning rates
    weight_decays=None, #and all weight decays
    min_or_mean='min', #use the minimum/best RMSE 
    include_persistence=True, #compare against the persistence model
    include_epochs=True, #to show how long each model took until covergence
    include_hyperparams=True) #show which hyperparams performed the best

print('This is the best RMSE (in W/m^2) and the best performing hyperparameter '
        'combination for every datatatype trained on 3 different forecast horizons: \n')
print(rmse_of_hyperparams)


################################################################################
############# (4B.4) PLOT BEST TRAIN AND VAL ERROR PER DATATYPE AND HORIZON 
################################################################################

# Let's plot the train and validation RMSEs of the best performing moodel
# i.e. the model with the optimal delta-length-hyperparameter combination
# for each datatype and horizon per epoch.
# For this, we first need to get the best performing deltas, lengths, and hyperparams.

# Get the deltas, lengths, and hyperparams for the horizon y_30.
_, delta_y30_img, length_y30_img, hyper_y30_img = getperformance.get_best_model_by_horizon(
    datatype='images', horizon=30, return_delta_length=True, return_hyperparams=True,
    batch_size=None, lr=None, weight_decay=None) #Search the best across all hyperparams

_, delta_y30_img3d, length_y30_img3d, hyper_y30_img3d = getperformance.get_best_model_by_horizon(
    datatype='images3d', horizon=30, return_delta_length=True, return_hyperparams=True,
    batch_size=None, lr=None, weight_decay=None) 

_, delta_y30_irr, length_y30_irr, hyper_y30_irr = getperformance.get_best_model_by_horizon(
    datatype='irradiance', horizon=30, return_delta_length=True, return_hyperparams=True,
    batch_size=None, lr=None, weight_decay=None) 

_, delta_y30_weath, length_y30_weath, hyper_y30_weath = getperformance.get_best_model_by_horizon(
    datatype='weather', horizon=30, return_delta_length=True, return_hyperparams=True,
    batch_size=None, lr=None, weight_decay=None) 

_, delta_y30_comb, length_y30_comb, hyper_y30_comb = getperformance.get_best_model_by_horizon(
    datatype='combined', horizon=30, return_delta_length=True, return_hyperparams=True,
    batch_size=None, lr=None, weight_decay=None) 

# Get the deltas, lengths, and hyperparams for the horizon y_120.
_, delta_y120_img, length_y120_img, hyper_y120_img = getperformance.get_best_model_by_horizon(
    datatype='images', horizon=120, return_delta_length=True, return_hyperparams=True,
    batch_size=None, lr=None, weight_decay=None) #Search the best across all hyperparams

_, delta_y120_img3d, length_y120_img3d, hyper_y120_img3d = getperformance.get_best_model_by_horizon(
    datatype='images3d', horizon=120, return_delta_length=True, return_hyperparams=True,
    batch_size=None, lr=None, weight_decay=None) 

_, delta_y120_irr, length_y120_irr, hyper_y120_irr = getperformance.get_best_model_by_horizon(
    datatype='irradiance', horizon=120, return_delta_length=True, return_hyperparams=True,
    batch_size=None, lr=None, weight_decay=None) 

_, delta_y120_weath, length_y120_weath, hyper_y120_weath = getperformance.get_best_model_by_horizon(
    datatype='weather', horizon=120, return_delta_length=True, return_hyperparams=True,
    batch_size=None, lr=None, weight_decay=None) 

_, delta_y120_comb, length_y120_comb, hyper_y120_comb = getperformance.get_best_model_by_horizon(
    datatype='combined', horizon=120, return_delta_length=True, return_hyperparams=True,
    batch_size=None, lr=None, weight_decay=None)

# Get the deltas, lengths, and hyperparams for the horizon y_1440.
_, delta_y1440_img, length_y1440_img, hyper_y1440_img = getperformance.get_best_model_by_horizon(
    datatype='images', horizon=1440, return_delta_length=True, return_hyperparams=True,
    batch_size=None, lr=None, weight_decay=None) #Search the best across all hyperparams

_, delta_y1440_img3d, length_y1440_img3d, hyper_y1440_img3d = getperformance.get_best_model_by_horizon(
    datatype='images3d', horizon=1440, return_delta_length=True, return_hyperparams=True,
    batch_size=None, lr=None, weight_decay=None)

_, delta_y1440_irr, length_y1440_irr, hyper_y1440_irr = getperformance.get_best_model_by_horizon(
    datatype='irradiance', horizon=1440, return_delta_length=True, return_hyperparams=True,
    batch_size=None, lr=None, weight_decay=None)

_, delta_y1440_weath, length_y1440_weath, hyper_y1440_weath = getperformance.get_best_model_by_horizon(
    datatype='weather', horizon=1440, return_delta_length=True, return_hyperparams=True,
    batch_size=None, lr=None, weight_decay=None)

_, delta_y1440_comb, length_y1440_comb, hyper_y1440_comb = getperformance.get_best_model_by_horizon(
    datatype='combined', horizon=1440, return_delta_length=True, return_hyperparams=True,
    batch_size=None, lr=None, weight_decay=None)

# We can now use these values to plot the train and validation mini-batch RMSEs.
# Plot train and val errors of the horizon y_30:
if PLOT_GRAPHS == True:

    # Plot losses of the best layer architecture per datatype.
    plotting.plot_best_train_val(
        datatypes=all_datatypes,
        deltas=[
            delta_y30_img, #best performing delta for this datatype and horizon
            delta_y30_img3d, 
            delta_y30_irr, 
            delta_y30_weath, 
            delta_y30_comb],
        lengths=[
            length_y30_img, #best performing length for this datatype and horizon
            length_y30_img3d, 
            length_y30_irr, 
            length_y30_weath, 
            length_y30_comb],
        horizon=30,
        batch_sizes=[
            hyper_y30_img['batch_size'], #best performing batch size
            hyper_y30_img3d['batch_size'],
            hyper_y30_irr['batch_size'],
            hyper_y30_weath['batch_size'],
            hyper_y30_comb['batch_size']],
        lrs=[
            hyper_y30_img['lr'], #best performing learning rate
            hyper_y30_img3d['lr'],
            hyper_y30_irr['lr'],
            hyper_y30_weath['lr'],
            hyper_y30_comb['lr']],
        weight_decays=[
            hyper_y30_img['weight_decay'], #best performing weight decay
            hyper_y30_img3d['weight_decay'],
            hyper_y30_irr['weight_decay'],
            hyper_y30_weath['weight_decay'],
            hyper_y30_comb['weight_decay']])

# Plot train and val errors of the horizon y_120:
if PLOT_GRAPHS == True:

    # Plot losses of the best layer architecture per datatype.
    plotting.plot_best_train_val(
        datatypes=all_datatypes,
        deltas=[
            delta_y120_img, #best performing delta for this datatype and horizon
            delta_y120_img3d, 
            delta_y120_irr, 
            delta_y120_weath, 
            delta_y120_comb],
        lengths=[
            length_y120_img, #best performing length for this datatype and horizon
            length_y120_img3d, 
            length_y120_irr, 
            length_y120_weath, 
            length_y120_comb],
        horizon=120,
        batch_sizes=[
            hyper_y120_img['batch_size'], #best performing batch size
            hyper_y120_img3d['batch_size'],
            hyper_y120_irr['batch_size'],
            hyper_y120_weath['batch_size'],
            hyper_y120_comb['batch_size']],
        lrs=[
            hyper_y120_img['lr'], #best performing learning rate
            hyper_y120_img3d['lr'],
            hyper_y120_irr['lr'],
            hyper_y120_weath['lr'],
            hyper_y120_comb['lr']],
        weight_decays=[
            hyper_y120_img['weight_decay'], #best performing weight decay
            hyper_y120_img3d['weight_decay'],
            hyper_y120_irr['weight_decay'],
            hyper_y120_weath['weight_decay'],
            hyper_y120_comb['weight_decay']])

# Plot train and val errors of the horizon y_1440:
if PLOT_GRAPHS == True:

    # Plot losses of the best layer architecture per datatype.
    plotting.plot_best_train_val(
        datatypes=all_datatypes,
        deltas=[
            delta_y1440_img, #best performing delta for this datatype and horizon
            delta_y1440_img3d, 
            delta_y1440_irr, 
            delta_y1440_weath, 
            delta_y1440_comb],
        lengths=[
            length_y1440_img, #best performing length for this datatype and horizon
            length_y1440_img3d, 
            length_y1440_irr, 
            length_y1440_weath, 
            length_y1440_comb],
        horizon=1440,
        batch_sizes=[
            hyper_y1440_img['batch_size'], #best performing batch size
            hyper_y1440_img3d['batch_size'],
            hyper_y1440_irr['batch_size'],
            hyper_y1440_weath['batch_size'],
            hyper_y1440_comb['batch_size']],
        lrs=[
            hyper_y1440_img['lr'], #best performing learning rate
            hyper_y1440_img3d['lr'],
            hyper_y1440_irr['lr'],
            hyper_y1440_weath['lr'],
            hyper_y1440_comb['lr']],
        weight_decays=[
            hyper_y1440_img['weight_decay'], #best performing weight decay
            hyper_y1440_img3d['weight_decay'],
            hyper_y1440_irr['weight_decay'],
            hyper_y1440_weath['weight_decay'],
            hyper_y1440_comb['weight_decay']])


################################################################################
############# (4B.5) COMPARE RMSE ACROSS DIFFERENT HYPERPARAMETERS
################################################################################

if PLOT_GRAPHS == True:
    
    # Plot the performance of the CNN layers with image data.
    plotting.plot_performance_by_hyperparam(
        datatypes=all_datatypes,
        horizons=horizon_buckets,
        min_or_mean='mean')


"""
################################################################################
############# 
############# (5A) FINAL TRAINING ON ALL HORIZONS
############# 
################################################################################
"""

# We now have one model trained with a specific delta-length-hyperparameter
# combination for each datatype and each forecast horizon bucket, i.e.
# intra-hour, intra-day, and day-ahead. We can now use each of these combinations
# to train each model on the other forecast horizons in the respective bucket.
# For this, we can use the already extracted best performing deltas, lengths,
# and hyperparameters per datatype and per horizon (we extracted those in (4B.2)
# in order to plot the best train and val mini-batch RMSEs per epoch).

# We already have our list of all the forecast horizons we want to consider.
all_horizons = [15, 30, 45, 60, 2*60, 3*60, 6*60, 12*60, 24*60, 2*24*60, 3*24*60]

# We will now define a list for each forecast horizon bucket.
intra_hours = [i for i in all_horizons if i < 60] #everything < 1 hour
intra_days = [i for i in all_horizons if i >=60 and i < 24*60] #everything < 1 day
day_aheads = [i for i in all_horizons if i >= 24*60] #everything > 1 day


################################################################################
############# (5A.1) TRAINING THE BEST IMAGES MODELS ON ALL HORIZONS
################################################################################

# Training the CNN with images data and all intra-hour horizons:
if TRAIN_MODELS == True:

    models.train_multiple_models(
        modeltype='CNN',
        datatype='images',
        sample=True, #otherwise GPU runs out of memory
        sample_size=0.02,
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_images], #the parameters of the best architecture
        delta_list=[delta_y30_img], #best performing delta for this horizon bucket
        length_list=[length_y30_img], #best performing length for this horizon bucket
        horizon_list=intra_hours, #all horizons in this bucket
        hyperparameter_dict={
            'batch_size':[hyper_y30_img['batch_size']], #best performing batch size
            'lr':[hyper_y30_img['lr']], #best performing learning rate
            'weight_decay':[hyper_y30_img['weight_decay']]}, #best weight decay
        loss_fn=models.RMSELoss(),
        layers_only=False, 
        early_stopping=True, #will stop training at convergence
        from_scratch=False)
        
# Training the CNN with images data and all intra-day horizons:
if TRAIN_MODELS == True:

    models.train_multiple_models(
        modeltype='CNN',
        datatype='images',
        sample=True, #otherwise GPU runs out of memory
        sample_size=0.02,
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_images], #the parameters of the best architecture
        delta_list=[delta_y120_img], #best performing delta for this horizon bucket
        length_list=[length_y120_img], #best performing length for this horizon bucket
        horizon_list=intra_days, #all horizons in this bucket
        hyperparameter_dict={
            'batch_size':[hyper_y120_img['batch_size']], #best performing batch size
            'lr':[hyper_y120_img['lr']], #best performing learning rate
            'weight_decay':[hyper_y120_img['weight_decay']]}, #best weight decay
        loss_fn=models.RMSELoss(),
        layers_only=False, 
        early_stopping=True, #will stop training at convergence
        from_scratch=False)

# Training the CNN with images data and all day-ahead horizons:
if TRAIN_MODELS == True:

    models.train_multiple_models(
        modeltype='CNN',
        datatype='images',
        sample=True, #otherwise GPU runs out of memory
        sample_size=0.02,
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_images], #the parameters of the best architecture
        delta_list=[delta_y1440_img], #best performing delta for this horizon bucket
        length_list=[length_y1440_img], #best performing length for this horizon bucket
        horizon_list=day_aheads, #all horizons in this bucket
        hyperparameter_dict={
            'batch_size':[hyper_y1440_img['batch_size']], #best performing batch size
            'lr':[hyper_y1440_img['lr']], #best performing learning rate
            'weight_decay':[hyper_y1440_img['weight_decay']]}, #best weight decay
        loss_fn=models.RMSELoss(),
        layers_only=False, 
        early_stopping=True, #will stop training at convergence
        from_scratch=False)
        

################################################################################
############# (5A.2) TRAINING THE BEST IMAGES3D MODELS ON ALL HORIZONS
################################################################################

# Training the 3D CNN with images3d data and all intra-hour horizons:
if TRAIN_MODELS == True:

    models.train_multiple_models(
        modeltype='CNN',
        datatype='images3d',
        sample=True, #otherwise GPU runs out of memory
        sample_size=0.02,
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_images3d], #the parameters of the best architecture
        delta_list=[delta_y30_img3d], #best performing delta for this horizon bucket
        length_list=[length_y30_img3d], #best performing length for this horizon bucket
        horizon_list=intra_hours, #all horizons in this bucket
        hyperparameter_dict={
            'batch_size':[hyper_y30_img3d['batch_size']], #best performing batch size
            'lr':[hyper_y30_img3d['lr']], #best performing learning rate
            'weight_decay':[hyper_y30_img3d['weight_decay']]}, #best weight decay
        loss_fn=models.RMSELoss(),
        layers_only=False, 
        early_stopping=True, #will stop training at convergence
        from_scratch=False)
        
# Training the 3D CNN with images3d data and all intra-day horizons:
if TRAIN_MODELS == True:

    models.train_multiple_models(
        modeltype='CNN',
        datatype='images3d',
        sample=True, #otherwise GPU runs out of memory
        sample_size=0.02,
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_images3d], #the parameters of the best architecture
        delta_list=[delta_y120_img3d], #best performing delta for this horizon bucket
        length_list=[length_y120_img3d], #best performing length for this horizon bucket
        horizon_list=intra_days, #all horizons in this bucket
        hyperparameter_dict={
            'batch_size':[hyper_y120_img3d['batch_size']], #best performing batch size
            'lr':[hyper_y120_img3d['lr']], #best performing learning rate
            'weight_decay':[hyper_y120_img3d['weight_decay']]}, #best weight decay
        loss_fn=models.RMSELoss(),
        layers_only=False, 
        early_stopping=True, #will stop training at convergence
        from_scratch=False)

# Training the 3D CNN with images3d data and all day-ahead horizons:
if TRAIN_MODELS == True:

    models.train_multiple_models(
        modeltype='CNN',
        datatype='images3d',
        sample=True, #otherwise GPU runs out of memory
        sample_size=0.02,
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_images3d], #the parameters of the best architecture
        delta_list=[delta_y1440_img3d], #best performing delta for this horizon bucket
        length_list=[length_y1440_img3d], #best performing length for this horizon bucket
        horizon_list=day_aheads, #all horizons in this bucket
        hyperparameter_dict={
            'batch_size':[hyper_y1440_img3d['batch_size']], #best performing batch size
            'lr':[hyper_y1440_img3d['lr']], #best performing learning rate
            'weight_decay':[hyper_y1440_img3d['weight_decay']]}, #best weight decay
        loss_fn=models.RMSELoss(),
        layers_only=False, 
        early_stopping=True, #will stop training at convergence
        from_scratch=False)
        

################################################################################
############# (5A.3) TRAINING THE BEST IRRADIANCE MODELS ON ALL HORIZONS
################################################################################

# Training the LSTM with irradiance data and all intra-hour horizons:
if TRAIN_MODELS == True:

    models.train_multiple_models(
        modeltype='LSTM',
        datatype='irradiance',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_irradiance], #the parameters of the best architecture
        delta_list=[delta_y30_irr], #best performing delta for this horizon bucket
        length_list=[length_y30_irr], #best performing length for this horizon bucket
        horizon_list=intra_hours, #all horizons in this bucket
        hyperparameter_dict={
            'batch_size':[hyper_y30_irr['batch_size']], #best performing batch size
            'lr':[hyper_y30_irr['lr']], #best performing learning rate
            'weight_decay':[hyper_y30_irr['weight_decay']]}, #best weight decay
        loss_fn=models.RMSELoss(),
        layers_only=False, 
        early_stopping=True, #will stop training at convergence
        from_scratch=False) #we want to train this model from scratch with 100% of the data

# Training the LSTM with irradiance data and all intra-day horizons:
if TRAIN_MODELS == True:

    models.train_multiple_models(
        modeltype='LSTM',
        datatype='irradiance',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_irradiance], #the parameters of the best architecture
        delta_list=[delta_y120_irr], #best performing delta for this horizon bucket
        length_list=[length_y120_irr], #best performing length for this horizon bucket
        horizon_list=intra_days, #all horizons in this bucket
        hyperparameter_dict={
            'batch_size':[hyper_y120_irr['batch_size']], #best performing batch size
            'lr':[hyper_y120_irr['lr']], #best performing learning rate
            'weight_decay':[hyper_y120_irr['weight_decay']]}, #best weight decay
        loss_fn=models.RMSELoss(),
        layers_only=False, 
        early_stopping=True, #will stop training at convergence
        from_scratch=False) #we want to train this model from scratch with 100% of the data

# Training the LSTM with irradiance data and all day-ahead horizons:
if TRAIN_MODELS == True:

    models.train_multiple_models(
        modeltype='LSTM',
        datatype='irradiance',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_irradiance], #the parameters of the best architecture
        delta_list=[delta_y1440_irr], #best performing delta for this horizon bucket
        length_list=[length_y1440_irr], #best performing length for this horizon bucket
        horizon_list=day_aheads, #all horizons in this bucket
        hyperparameter_dict={
            'batch_size':[hyper_y1440_irr['batch_size']], #best performing batch size
            'lr':[hyper_y1440_irr['lr']], #best performing learning rate
            'weight_decay':[hyper_y1440_irr['weight_decay']]}, #best weight decay
        loss_fn=models.RMSELoss(),
        layers_only=False, 
        early_stopping=True, #will stop training at convergence
        from_scratch=False) #we want to train this model from scratch with 100% of the data


################################################################################
############# (5A.4) TRAINING THE BEST WEATHER MODELS ON ALL HORIZONS
################################################################################

# Training the LSTM with weather data and all intra-hour horizons:
if TRAIN_MODELS == True:

    models.train_multiple_models(
        modeltype='LSTM',
        datatype='weather',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_weather], #the parameters of the best architecture
        delta_list=[delta_y30_weath], #best performing delta for this horizon bucket
        length_list=[length_y30_weath], #best performing length for this horizon bucket
        horizon_list=intra_hours, #all horizons in this bucket
        hyperparameter_dict={
            'batch_size':[hyper_y30_weath['batch_size']], #best performing batch size
            'lr':[hyper_y30_weath['lr']], #best performing learning rate
            'weight_decay':[hyper_y30_weath['weight_decay']]}, #best weight decay
        loss_fn=models.RMSELoss(),
        layers_only=False, 
        early_stopping=True, #will stop training at convergence
        from_scratch=False) #we want to train this model from scratch with 100% of the data

# Training the LSTM with weather data and all intra-day horizons:
if TRAIN_MODELS == True:

    models.train_multiple_models(
        modeltype='LSTM',
        datatype='weather',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_weather], #the parameters of the best architecture
        delta_list=[delta_y120_weath], #best performing delta for this horizon bucket
        length_list=[length_y120_weath], #best performing length for this horizon bucket
        horizon_list=intra_days, #all horizons in this bucket
        hyperparameter_dict={
            'batch_size':[hyper_y120_weath['batch_size']], #best performing batch size
            'lr':[hyper_y120_weath['lr']], #best performing learning rate
            'weight_decay':[hyper_y120_weath['weight_decay']]}, #best weight decay
        loss_fn=models.RMSELoss(),
        layers_only=False, 
        early_stopping=True, #will stop training at convergence
        from_scratch=False) #we want to train this model from scratch with 100% of the data

# Training the LSTM with weather data and all day-ahead horizons:
if TRAIN_MODELS == True:

    models.train_multiple_models(
        modeltype='LSTM',
        datatype='weather',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_weather], #the parameters of the best architecture
        delta_list=[delta_y1440_weath], #best performing delta for this horizon bucket
        length_list=[length_y1440_weath], #best performing length for this horizon bucket
        horizon_list=day_aheads, #all horizons in this bucket
        hyperparameter_dict={
            'batch_size':[hyper_y1440_weath['batch_size']], #best performing batch size
            'lr':[hyper_y1440_weath['lr']], #best performing learning rate
            'weight_decay':[hyper_y1440_weath['weight_decay']]}, #best weight decay
        loss_fn=models.RMSELoss(),
        layers_only=False, 
        early_stopping=True, #will stop training at convergence
        from_scratch=False) #we want to train this model from scratch with 100% of the data


################################################################################
############# (5A.5) TRAINING THE BEST COMBINED MODELS ON ALL HORIZONS
################################################################################

# Training the LSTM with combined data and all intra-hour horizons:
if TRAIN_MODELS == True:

    models.train_multiple_models(
        modeltype='LSTM',
        datatype='combined',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_combined], #the parameters of the best architecture
        delta_list=[delta_y30_comb], #best performing delta for this horizon bucket
        length_list=[length_y30_comb], #best performing length for this horizon bucket
        horizon_list=intra_hours, #all horizons in this bucket
        hyperparameter_dict={
            'batch_size':[hyper_y30_comb['batch_size']], #best performing batch size
            'lr':[hyper_y30_comb['lr']], #best performing learning rate
            'weight_decay':[hyper_y30_comb['weight_decay']]}, #best weight decay
        loss_fn=models.RMSELoss(),
        layers_only=False, 
        early_stopping=True, #will stop training at convergence
        from_scratch=False) #we want to train this model from scratch with 100% of the data

# Training the LSTM with combined data and all intra-day horizons:
if TRAIN_MODELS == True:

    models.train_multiple_models(
        modeltype='LSTM',
        datatype='combined',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_combined], #the parameters of the best architecture
        delta_list=[delta_y120_comb], #best performing delta for this horizon bucket
        length_list=[length_y120_comb], #best performing length for this horizon bucket
        horizon_list=intra_days, #all horizons in this bucket
        hyperparameter_dict={
            'batch_size':[hyper_y120_comb['batch_size']], #best performing batch size
            'lr':[hyper_y120_comb['lr']], #best performing learning rate
            'weight_decay':[hyper_y120_comb['weight_decay']]}, #best weight decay
        loss_fn=models.RMSELoss(),
        layers_only=False, 
        early_stopping=True, #will stop training at convergence
        from_scratch=False) #we want to train this model from scratch with 100% of the data

# Training the LSTM with combined data and all day-ahead horizons:
if TRAIN_MODELS == True:

    models.train_multiple_models(
        modeltype='LSTM',
        datatype='combined',
        sample=True, #otherwise training takes too long
        sample_size='7min',
        GPU=True,
        n_epochs=500, #large value to ensure convergence
        layers_list=[best_layers_combined], #the parameters of the best architecture
        delta_list=[delta_y1440_comb], #best performing delta for this horizon bucket
        length_list=[length_y1440_comb], #best performing length for this horizon bucket
        horizon_list=day_aheads, #all horizons in this bucket
        hyperparameter_dict={
            'batch_size':[hyper_y1440_comb['batch_size']], #best performing batch size
            'lr':[hyper_y1440_comb['lr']], #best performing learning rate
            'weight_decay':[hyper_y1440_comb['weight_decay']]}, #best weight decay
        loss_fn=models.RMSELoss(),
        layers_only=False, 
        early_stopping=True, #will stop training at convergence
        from_scratch=False) #we want to train this model from scratch with 100% of the data


"""
################################################################################
############# 
############# (5B) ANALYZING THE PERFORMANCE OF EACH FINAL HORIZON
############# 
################################################################################
"""

# Again, these are the datatypes we trained the models with
# and ALL the forecast horizons we now trained the models on:
all_datatypes = ['images', 'images3d', 'irradiance', 'weather', 'combined']
all_horizons = [15, 30, 45, 60, 2*60, 3*60, 6*60, 12*60, 24*60, 2*24*60, 3*24*60]

################################################################################
############# (5B.1) PLOT VALIDATION ERRORS PER DATATYPE AND HORIZON
################################################################################

# Plot all horizons vs. RMSE by datatype.
if PLOT_GRAPHS == True:

    plotting.plot_rmse_by_horizon(
        datatypes=all_datatypes,
        horizons=all_horizons)


"""
################################################################################
############# 
############# (6) ANALYZING THE ERRORS
############# 
################################################################################
"""

# In order to make subsequent analyses and trainings faster,
# let's utilize the best model for each datatype-horizon combination
# and make predictions on the train, val, and test set for each.
# We will save these predictions and their timestamps for subsequent
# analyses and trainings.

# Note that this can take a while.
if DATA_FROM_SCRATCH == True:

    data.create_prediction_data(
        datatype='images', horizons=all_horizons)
    data.create_prediction_data(
        datatype='images3d', horizons=all_horizons)
    data.create_prediction_data(
        datatype='irradiance', horizons=all_horizons)
    data.create_prediction_data(
        datatype='weather', horizons=all_horizons)
    data.create_prediction_data(
        datatype='combined', horizons=all_horizons)


################################################################################
############# (6.1) COMPARE MEAN MINI-BATCH RMSE VS. FULL DATA RMSE
################################################################################

# All previous analyses analyzed the average mini-batch RMSEs that were computed
# during training and where fed into the optimizer to tune the model parameters.
# However, this was based on a) mini-batches rather than on the whole dataset 
# and b) on a sample of the data. Let's compute the validation RMSE for each datatype
# and horizon bucket on the WHOLE dataset and compare it to the mean mini-batch
# RMSE used during training.
# We will compare this by denoting the %-change from mean mini-batch val. RMSE to
# the whole dataset val. RMSE.
# We will also count the number of models trained per datatype.

# Compare whole dataset val. RMSE vs. mean mini-batch val. RMSE.
full_vs_mini_rmse = getperformance.compare_final_rmses(
    datatypes=all_datatypes,
    horizons=horizon_buckets)

print('This is a comparison between the full vs. mini-batch validation '
    'RMSEs for each datatype and horizon bucket: \n')
print(full_vs_mini_rmse)


################################################################################
############# (6.2) PLOT FULL-DATA VALIDATION ERRORS PER DATATYPE AND HORIZON
################################################################################

# Given that the distinction between full-data validation RMSEs and 
# mean mini-batch validatin RMSE is material, let's look at the plot
# of RMSE vs. horizon per model again, but this time using the FULL-DATA RMSE.

# Plot FULL-DATA validation RMSE per datatype and horizon.
if PLOT_GRAPHS == True:

    plotting.plot_rmse_by_horizon(
        datatypes=all_datatypes,
        horizons=all_horizons,
        full_data=True)

################################################################################
############# (6.3) ERRORS BY TIME
################################################################################

# Plot the errors vs. time of the day.
if PLOT_GRAPHS == True:
        
    plotting.plot_errors_by_metric(
        metric='time',
        sample_size=0.001, #to make scatterplot readable
        datatypes=all_datatypes,
        horizons=horizon_buckets, #30mins, 2hours, 1day horizons
        which_set='val') #plot the errors on the validation set

# Plot the errors vs. month of the year.
if PLOT_GRAPHS == True:
        
    plotting.plot_errors_by_metric(
        metric='month',
        sample_size=0.001,
        datatypes=all_datatypes,
        horizons=horizon_buckets,
        which_set='val')

# Also plot the errors vs. time, but on the train set.
# This can show us potential areas of overfit.
if PLOT_GRAPHS == True:
        
    plotting.plot_errors_by_metric(
        metric='time',
        sample_size=0.001, #to make scatterplot readable
        datatypes=all_datatypes,
        horizons=horizon_buckets, #30mins, 2hours, 1day horizons
        which_set='train') #plot the errors on the validation set


################################################################################
############# (6.4) ERRORS BY GHI
################################################################################

# Plot the errors vs. irradiance level
if PLOT_GRAPHS == True:
        
    plotting.plot_errors_by_metric(
        metric='irradiance',
        sample_size=0.001,
        datatypes=all_datatypes,
        horizons=horizon_buckets,
        which_set='val')


################################################################################
############# (6.5) ERRORS BY WEATHER
################################################################################

# Plot the errors vs. temperature.
if PLOT_GRAPHS == True:
        
    plotting.plot_errors_by_metric(
        metric='air_temp',
        sample_size=0.001,
        datatypes=all_datatypes,
        horizons=horizon_buckets,
        which_set='val')

# Plot the errors vs. relative humidity.
if PLOT_GRAPHS == True:
        
    plotting.plot_errors_by_metric(
        metric='relhum',
        sample_size=0.001,
        datatypes=all_datatypes,
        horizons=horizon_buckets,
        which_set='val')

# Plot the errors vs. pressure.
if PLOT_GRAPHS == True:
        
    plotting.plot_errors_by_metric(
        metric='press',
        sample_size=0.001,
        datatypes=all_datatypes,
        horizons=horizon_buckets,
        which_set='val')

# Plot the errors vs. wind speed.
if PLOT_GRAPHS == True:
        
    plotting.plot_errors_by_metric(
        metric='windsp',
        sample_size=0.001,
        datatypes=all_datatypes,
        horizons=horizon_buckets,
        which_set='val')

# Plot the errors vs. wind direction.
if PLOT_GRAPHS == True:
        
    plotting.plot_errors_by_metric(
        metric='winddir',
        sample_size=0.001,
        datatypes=all_datatypes,
        horizons=horizon_buckets,
        which_set='val')


"""
################################################################################
#############
############# (7) ENSEMBLE MODELS: LINEAR COMBINATIONS
#############
################################################################################
"""

# We now want to use the predictions of each individual model and combine them
# to train one ensemble model that hopefully reduces the full-data validation RMSE.
# For this, we will first train linear combination models to get a first baseline
# that will indicate whether this approach has any merit.


################################################################################
############# (7.1) COMPARE THE LINEAR BASELINE ACROSS ALL ENSEMBLE DATA 
################################################################################

# Let us compare the full-data train and validation RMSEs for each different combination
# of ensemble data. For example, one model will be trained on just the individual
# model predictions, another one will also include the interaction terms, another will
# also include time data, etc.
# Let's compare these RMSEs for our three forecast horizon buckets.
linear_rmses = getperformance.compare_linear_combinations(
    horizons=horizon_buckets)

print('This is full-data train and validation RMSE (in W/m^2) for each '
    'linear ensemble data combination and per horizon bucket: \n')
print(linear_rmses)


################################################################################
############# (7.2) BEST LINEAR COMBINATIONS VS. BEST INDIVIDUAL MODELS BY HORIZON
################################################################################

# Let us now repeat this linear combination fitting, but this time for all
# forecast horizons, not just the three buckets. Then, compare the best performing
# linear combination model to the best performing individual model (e.g. CNN with
# image data) and the persistence model for each horizon.

if PLOT_GRAPHS == True:

    plotting.plot_linear_combination_vs_indiv(
        horizons=all_horizons)


################################################################################
############# (7.3) EXAMINE LINEAR COEFFICIENTS
################################################################################

# Given that the prediction+time linear combination model performed quite well,
# let's examine the linear coefficients of this model for all three
# forecast horizon buckets.
coef_df = getperformance.compare_linear_coef(
    horizons=horizon_buckets,
    include_time=True) #we want to look at the prediction+time model

print('These are the coefficients of the linear combination model '
    'trained on the predictions+time ensemble data for the '
    'three forecasting buckets: \n')
print(coef_df)


################################################################################
############# (7.4) PLOT LINEAR COEFFICIENTS PER HORIZON
################################################################################

# Let's now plot these coefficients for all forecast horizons.
if PLOT_GRAPHS == True:

    plotting.plot_linear_coef(
    horizons=all_horizons)


"""
################################################################################
#############
############# (8) ENSEMBLE MODELS: NEURAL NETWORKS - TRAIN DIFFERENT ARCHITECTURES
#############
################################################################################
"""

################################################################################
############# (8.1) TRAIN DIFFERENT ENSEMBLE LAYER ARCHITECTURES
################################################################################

# Our goal is to first test different layer architectures to find the best
# performing architecture. For this, we first need to define a list of all 
# possible layer architectures that can be used to create our models.NN() class.
# We will use the following values for the different parts of the NN architecture:

all_hidden_num = [2, 3, 4, 5] #we will train NNs with 2, 3, 4 hidden layers
all_hidden_size = [100, 250, 500, 750] #these are the different hidden layer sizes
all_dropout_prob = [0.3, 0.5, 0.7] #these are the dropout probabilities we'll check

# We can use these to create a combination of all different layer values.
nn_layers_list = models.create_NN_combinations(
    all_hidden_num=all_hidden_num,
    all_hidden_size=all_hidden_size,
    all_dropout_prob=all_dropout_prob)

# Lastly, we need to define a dictionary of hyperparameters we want to use 
# to train these different ensemble layer architectures.
# We'll use the same hyperparameters as for the training of the individual
# layer architectures in (2A):
hyperparams = {'batch_size':[128],
                'lr':[1e-04],
                'weight_decay':[1e-05]}

# We can now train one NN ensemble model for each of these layer architectures.
if TRAIN_MODELS == True:

    models.train_multiple_ensemble_models(
        GPU=True, #use the GPU to train on
        layers_list=nn_layers_list, #all architectures we just created
        horizon_list=[120], #we'll train all on a 2-hour ahead forecast, similar to (2A)
        hyperparameter_dict=hyperparams,
        all_inputs=False,
        from_scratch=False) #do not train from scratch if training was interrupted


################################################################################
############# (8.2) GET PERFORMANCE OVERVIEW OF NN ENSEMBLE ARCHITECTURES
################################################################################

# Let's print the min, mean, and std. of the full validation RMSEs
# across all of the trained NN ensemble architectures.
# This will give us a feel of potential model uncertainty in this step of the
# whole training process.
ensemble_perf_of_layers = getperformance.get_ensemble_performance_of_layers()

print('This is the min., mean, and std. of the full validation RMSEs '
    'of the NN ensemble models across their trained layer architectures: \n')
print(ensemble_perf_of_layers)


################################################################################
############# (8.3) PLOT THESE DIFFERENT ENSEMBLE ARCHITECTURE PERFORMANCES
################################################################################

# Plot a histogram of the full validation RMSEs of all these architectures.
if PLOT_GRAPHS == True:

    plotting.plot_ensemble_layers_hist(
        horizon=120, #all of them were trained on a 2-hour ahead forecast
        batch_size=128, #and used these hyperparameters
        lr=0.0001,
        weight_decay=1e-05)


################################################################################
############# (8.4) CHECK BEST PERFORMING ENSEMBLE ARCHITECTURE
################################################################################

# Let's check which of the ~50 NN architetectures performed the best.
best_ensemble_layers = getperformance.get_best_ensemble_layers(
    batch_size=128, #we used these hyperparams in step (8)
    lr=0.0001,
    weight_decay=1e-05)
print('These are the parameters of the best-performing NN ensemble architecture: \n')
print(best_ensemble_layers)


"""
################################################################################
#############
############# (9) ENSEMBLE MODELS: NEURAL NETWORKS - TRAIN DIFFERENT INPUTS PER HORIZON
#############
################################################################################
"""

# We now want to see the impact of the different ensemble input combinations
# on the performance of the best performing architecture per horizon bucket.
# For this, we first need to get the best performing layer architecture.
best_ensemble_layers = getperformance.get_best_ensemble_layers(
    batch_size=128, #we used these hyperparams in step (8)
    lr=0.0001,
    weight_decay=1e-05)

# We will again train on the previously defined horizon buckets.
horizon_buckets = [30, 120, 60*24] #30mins, 2hours, 1day


################################################################################
############# (9.1) TRAIN DIFFERENT ENSEMBLE INPUT COMBINATIONS PER HORIZON BUCKET
################################################################################

# Train the best ensemble architecture on each ensemble data combination
# and horizon bucket.
if TRAIN_MODELS == True:
    models.train_multiple_ensemble_models(
        GPU=True, #use the GPU to train on
        layers_list=[best_ensemble_layers], #best ensemble architecture
        horizon_list=horizon_buckets, #30mins, 2hours, 1day
        hyperparameter_dict=hyperparams, #use same hyperparams as before
        all_inputs=True, #this time, use ALL ensemble data combinations
        from_scratch=False) #do not train from scratch if training was interrupted


################################################################################
############# (9.2) GET PERFORMANCE OVERVIEW OF NN ENSEMBLE INPUT COMBINATIONS
################################################################################

# Let's print the min, mean, and std. of the full validation RMSEs
# across all of the trained NN ensemble input combinations.
# This will give us a feel of potential model uncertainty in this step of the
# whole training process.

# Let's check it for y_30.
ensemble_perf_of_inputs_y30 = getperformance.get_ensemble_performance_of_inputs(
    horizon=30)
print('This is the min., mean, and std. of the full validation RMSEs '
    'of the NN ensemble models across all trained ensemble input combinations '
    'and for a forecast horizon of 30min: \n')
print(ensemble_perf_of_inputs_y30)

# Let's check it for y_120.
ensemble_perf_of_inputs_y120 = getperformance.get_ensemble_performance_of_inputs(
    horizon=120)
print('This is the min., mean, and std. of the full validation RMSEs '
    'of the NN ensemble models across all trained ensemble input combinations '
    'and for a forecast horizon of 120min: \n')
print(ensemble_perf_of_inputs_y120)

# Let's check it for y_1440
ensemble_perf_of_inputs_y1440 = getperformance.get_ensemble_performance_of_inputs(
    horizon=1440)
print('This is the min., mean, and std. of the full validation RMSEs '
    'of the NN ensemble models across all trained ensemble input combinations '
    'and for a forecast horizon of 1440min: \n')
print(ensemble_perf_of_inputs_y1440)


################################################################################
############# (9.3) COMPARE THE PERFORMANCE BY INPUT TO THE LINEAR BASELINE
################################################################################

# Get the full validation RMSE of the NN ensemble model for all inputs
# and get a %-difference to the linear baseline.
linear_vs_nn_alldata = getperformance.compare_linear_vs_nn(
    horizons=horizon_buckets, #compare against all horizon buckets
    batch_size=128, #the ensemble NNs used these hyperparams in this step
    lr=0.0001,
    weight_decay=1e-05)

print('This is a comparison between the NN ensemble model to the linear baseline '
        'for every ensemble input combination for all three horizon buckets: \n')
print(linear_vs_nn_alldata)


################################################################################
############# (9.4) PLOT THESE PERFORMANCE DIFFERENCES OF LINEAR VS NN ENSEMBLE MODELS
################################################################################

# Plot the above %-difference for each horizon as a histogram.
if PLOT_GRAPHS == True:

    plotting.plot_linear_vs_nn_hist(
        horizons=horizon_buckets, #compare against all horizon buckets
        batch_size=128, #the ensemble NNs used these hyperparams in this step
        lr=0.0001,
        weight_decay=1e-05)


################################################################################
############# (9.5) CHECK BEST PERFORMING ENSEMBLE INPUTS PER HORIZON
################################################################################

# Let's check which combination of ensemble inputs performed the best
# for the horizon y_30.
_, ensemble_inputs_y30 = getperformance.get_best_ensemble_model_by_horizon(
    horizon=30, return_inputs=True, return_hyperparams=False,
    batch_size=128, lr=1e-04, weight_decay=1e-05) #hyperparams used in this step (9)
print('These are the best-performing inputs for the 30-min-ahead NN ensemble: \n')
print(ensemble_inputs_y30)

# And for the horizon y_120.
_, ensemble_inputs_y120 = getperformance.get_best_ensemble_model_by_horizon(
    horizon=120, return_inputs=True, return_hyperparams=False,
    batch_size=128, lr=1e-04, weight_decay=1e-05) #hyperparams used in this step (9)
print('These are the best-performing inputs for the 2-hour-ahead NN ensemble: \n')
print(ensemble_inputs_y120)

# And for the horizon y_1440.
_, ensemble_inputs_y1440 = getperformance.get_best_ensemble_model_by_horizon(
    horizon=1440, return_inputs=True, return_hyperparams=False,
    batch_size=128, lr=1e-04, weight_decay=1e-05) #hyperparams used in this step (9)
print('These are the best-performing inputs for the 1-day-ahead NN ensemble: \n')
print(ensemble_inputs_y1440)


"""
################################################################################
#############
############# (10) ENSEMBLE MODELS: NEURAL NETWORKS - TUNE HYPERPARAMETERS
#############
################################################################################
"""

# We will span a similar space of hyperparams as with the individual models,
# however to reduce computational complexity, we will limit the grid-search
# to these sets of hyperparameters:
all_hyperparams_ensemble = {'batch_size':[64, 128],
                        'lr':[1e-04],
                        'weight_decay':[1e-05, 1e-04, 1e-03]}


################################################################################
############# (10.1) TUNE THE HYPERPARAMETERS OF THE ENSEMBLE MODELS PER BUCKET
################################################################################

# Tune the hyperparameters of the intra-hour ensemble model.
if TRAIN_MODELS == True:
    models.train_multiple_ensemble_models(
        GPU=True, #use the GPU to train on
        layers_list=[best_ensemble_layers], #best ensemble architecture
        horizon_list=[30], #only train this horizon bucket
        hyperparameter_dict=all_hyperparams_ensemble, #grid-search over these hyperparams
        all_inputs=False, #do NOT use all ensemble inputs, but instead..
        **ensemble_inputs_y30, #..use the best performing input combo per horizon bucket
        from_scratch=False) #do not train from scratch if training was interrupted

# Tune the hyperparameters of the intra-day ensemble model.
if TRAIN_MODELS == True:
    models.train_multiple_ensemble_models(
        GPU=True, #use the GPU to train on
        layers_list=[best_ensemble_layers], #best ensemble architecture
        horizon_list=[120], #only train this horizon bucket
        hyperparameter_dict=all_hyperparams_ensemble, #grid-search over these hyperparams
        all_inputs=False, #do NOT use all ensemble inputs, but instead..
        **ensemble_inputs_y120, #..use the best performing input combo per horizon bucket
        from_scratch=False) #do not train from scratch if training was interrupted

# Tune the hyperparameters of the day-ahead ensemble model.
if TRAIN_MODELS == True:
    models.train_multiple_ensemble_models(
        GPU=True, #use the GPU to train on
        layers_list=[best_ensemble_layers], #best ensemble architecture
        horizon_list=[1440], #only train this horizon bucket
        hyperparameter_dict=all_hyperparams_ensemble, #grid-search over these hyperparams
        all_inputs=False, #do NOT use all ensemble inputs, but instead..
        **ensemble_inputs_y1440, #..use the best performing input combo per horizon bucket
        from_scratch=False) #do not train from scratch if training was interrupted


################################################################################
############# (10.2) GET PERFORMANCE OVERVIEW OF NN ENSEMBLE HYPERPARAM COMBINATIONS
################################################################################

# Let's print the min, mean, and std. of the full validation RMSEs
# across all of the trained NN hyperparameter combinations.
# This will give us a feel of potential model uncertainty in this step of the
# whole training process.

# Let's check it for y_30.
ensemble_perf_of_hyperparams_y30 = getperformance.get_ensemble_performance_of_hyperparams(
    horizon=30)
print('This is the min., mean, and std. of the full validation RMSEs '
    'of the NN ensemble models across all trained hyperparameter combinations '
    'with a forecast horizon of 30min: \n')
print(ensemble_perf_of_hyperparams_y30)

# Let's check it for y_120.
ensemble_perf_of_hyperparams_y120 = getperformance.get_ensemble_performance_of_hyperparams(
    horizon=120)
print('This is the min., mean, and std. of the full validation RMSEs '
    'of the NN ensemble models across all trained hyperparameter combinations '
    'with a forecast horizon of 120min: \n')
print(ensemble_perf_of_hyperparams_y120)

# Let's check it for y_1440.
ensemble_perf_of_hyperparams_y1440 = getperformance.get_ensemble_performance_of_hyperparams(
    horizon=1440)
print('This is the min., mean, and std. of the full validation RMSEs '
    'of the NN ensemble models across all trained hyperparameter combinations '
    'with a forecast horizon of 1440min: \n')
print(ensemble_perf_of_hyperparams_y1440)


################################################################################
############# (10.3) CHECK BEST PERFORMING NN HYPERPARAMETERS PER HORIZON
################################################################################

# Check the best-performing hyperparameters for each horizon bucket.
# Check them for the horizon y_30.
_, ensemble_inputs_y30, ensemble_hyper_y30 = getperformance.get_best_ensemble_model_by_horizon(
    horizon=30, return_inputs=True, return_hyperparams=True, #return hyperparams
    batch_size=None, lr=None, weight_decay=None) #search across all hyperparams
print('These are the best-performing hyperparameters of the 30-min-ahead NN ensemble: \n')
print(ensemble_hyper_y30)

# And for the horizon y_120.
_, ensemble_inputs_y120, ensemble_hyper_y120 = getperformance.get_best_ensemble_model_by_horizon(
    horizon=120, return_inputs=True, return_hyperparams=True, #return hyperparams
    batch_size=None, lr=None, weight_decay=None) #search across all hyperparams
print('These are the best-performing hyperparameters of the 2-hour-ahead NN ensemble: \n')
print(ensemble_hyper_y120)

# And for the horizon y_1440.
_, ensemble_inputs_y1440, ensemble_hyper_y1440 = getperformance.get_best_ensemble_model_by_horizon(
    horizon=1440, return_inputs=True, return_hyperparams=True, #return hyperparams
    batch_size=None, lr=None, weight_decay=None) #search across all hyperparams
print('These are the best-performing hyperparameters of the 1-day-ahead NN ensemble: \n')
print(ensemble_hyper_y1440)


"""
################################################################################
#############
############# (11) ENSEMBLE MODELS: NEURAL NETWORKS - TRAIN ON ALL HORIZONS
#############
################################################################################
"""

# It is time for the last trainining step: for each forecast horizon bucket,
# we wanto train the model with the best performing inputs and hyperparameters
# on the remaining horizons in each bucket.
# For this, we need the best-performing hyperparameters that we found above.

# Also, we again, need the forecast horizons in each horizon bucket.
intra_hours = [i for i in all_horizons if i < 60] #everything < 1 hour
intra_days = [i for i in all_horizons if i >=60 and i < 24*60] #everything < 1 day
day_aheads = [i for i in all_horizons if i >= 24*60] #everything > 1 day


################################################################################
############# (11.1) TRAIN ON THE REMAINING FORECAST HORIZONS
################################################################################

# Train on the remaining intra-hour horizons.
if TRAIN_MODELS == True:

    models.train_multiple_ensemble_models(
        GPU=True, #use the GPU to train on
        layers_list=[best_ensemble_layers], #best ensemble architecture
        horizon_list=intra_hours, #only train horizons in this bucket
        hyperparameter_dict={ #best performing hyperparams of this bucket
            'batch_size':[ensemble_hyper_y30['batch_size']], #best performing batch size
            'lr':[ensemble_hyper_y30['lr']], #best performing learning rate
            'weight_decay':[ensemble_hyper_y30['weight_decay']]}, #best weight decay
        all_inputs=False, #do NOT use all ensemble inputs, but instead..
        **ensemble_inputs_y30, #..use the best performing input combo per horizon bucket
        from_scratch=False) #do not train from scratch if training was interrupted

# Train on the remaining intra-day horizons.
if TRAIN_MODELS == True:

    models.train_multiple_ensemble_models(
        GPU=True, #use the GPU to train on
        layers_list=[best_ensemble_layers], #best ensemble architecture
        horizon_list=intra_days, #only train horizons in this bucket
        hyperparameter_dict={ #best performing hyperparams of this bucket
            'batch_size':[ensemble_hyper_y120['batch_size']], #best performing batch size
            'lr':[ensemble_hyper_y120['lr']], #best performing learning rate
            'weight_decay':[ensemble_hyper_y120['weight_decay']]}, #best weight decay
        all_inputs=False, #do NOT use all ensemble inputs, but instead..
        **ensemble_inputs_y120, #..use the best performing input combo per horizon bucket
        from_scratch=False) #do not train from scratch if training was interrupted

# Train on the remaining day-ahead horizons.
if TRAIN_MODELS == True:

    models.train_multiple_ensemble_models(
        GPU=True, #use the GPU to train on
        layers_list=[best_ensemble_layers], #best ensemble architecture
        horizon_list=day_aheads, #only train horizons in this bucket
        hyperparameter_dict={ #best performing hyperparams of this bucket
            'batch_size':[ensemble_hyper_y1440['batch_size']], #best performing batch size
            'lr':[ensemble_hyper_y1440['lr']], #best performing learning rate
            'weight_decay':[ensemble_hyper_y1440['weight_decay']]}, #best weight decay
        all_inputs=False, #do NOT use all ensemble inputs, but instead..
        **ensemble_inputs_y1440, #..use the best performing input combo per horizon bucket
        from_scratch=False) #do not train from scratch if training was interrupted


################################################################################
############# (11.2) PLOT THE FULL VALIDATION RMSES FOR ALL MODELS BY HORIZON
################################################################################

# Plot the full validation RMSE for the individual and ensemble models for all horizons.
if PLOT_GRAPHS == True:
    
    print('Note that some of these data are computed during this process. '
        'The whole plotting process will, thus, take ~30mins. \n')

    plotting.plot_rmse_by_horizon(
        datatypes=all_datatypes,
        horizons=all_horizons,
        full_data=True, #use full validation RMSE
        include_ensemble=True) #also plot ensemble models


"""
################################################################################
#############
############# (12) SUMMARY
#############
################################################################################
"""

################################################################################
############# (12.1) SUMMARIZE ALL PERFORMANCE OVERVIEWS OF ALL MODEL TYPES
################################################################################

# Let's print the min, mean, and std. of the (mini-batch/full) validation RMSEs
# for all individual models and ensemble models across all training steps
# This will give us a feel of potential model uncertainty for each model
# and each part of the training

# Keep in mind that the individual models have their performance evaluated based
# the mini-batch validation RMSEs whereas the ensemble NNs use a full validation RMSEs.
# Also, note that the individual models where trained on a 1-hour ahead foreacast 
# in the 'layer architecture' step.

# Summarize performance per training step by model type for intra-hour horizon: 30min.
overall_perf_30 = getperformance.get_overall_performance(
    horizon=30)
print('This is the min., mean, and std. of the (mini-batch/full) validation RMSEs '
    'across all model types and training steps for the forecast horizon 30min: \n')
print(overall_perf_30)

# Summarize performance per training step by model type for intra-day horizon: 120min.
overall_perf_120 = getperformance.get_overall_performance(
    horizon=120)
print('This is the min., mean, and std. of the (mini-batch/full) validation RMSEs '
    'across all model types and training steps for the forecast horizon 120min: \n')
print(overall_perf_120)

# Summarize performance per training step by model type for day-ahead horizon: 1440min.
overall_perf_1440 = getperformance.get_overall_performance(
    horizon=1440)
print('This is the min., mean, and std. of the (mini-batch/full) validation RMSEs '
    'across all model types and training steps for the forecast horizon 1440min: \n')
print(overall_perf_1440)


################################################################################
############# (12.2) SUMMARIZE TRAIN, VAL, TEST RMSES OF ALL MODELS AND HORIZONS
################################################################################

# Let's summarize the full-data train, val, and test RMSEs across all model types.
# Print it for the intra-hour horizons.
overall_train_val_test_y30 = getperformance.get_overall_train_val_test_errors(
    horizons=intra_hours)
print('These are the full train, val, and test RMSEs per modeltype '
    'of the intra-hour horizons: \n')
print(overall_train_val_test_y30)

# And for the intra-day horizons.
overall_train_val_test_y120 = getperformance.get_overall_train_val_test_errors(
    horizons=intra_days)
print('These are the full train, val, and test RMSEs per modeltype '
    'of the intra-day horizons: \n')
print(overall_train_val_test_y120)

# And for the day-ahead horizons.
overall_train_val_test_y1440 = getperformance.get_overall_train_val_test_errors(
    horizons=day_aheads)
print('These are the full train, val, and test RMSEs per modeltype '
    'of the day-ahead horizons: \n')
print(overall_train_val_test_y1440)


################################################################################
############# (12.3) PLOT TRAIN, VAL, TEST %-DIFFERENCE OF INDIV. VS ENSEMBLE
################################################################################

# Let's plot the %-difference in train, val, test RMSEs between the best
# individiual and the best ensemble model per horizon.
if PLOT_GRAPHS == True:

    print('Plotting the percentage difference in RMSEs of the best '
        'indiviual vs. best ensemble models. This will take a while...')

    plotting.plot_train_val_test_difference_by_horizon(
        horizons=all_horizons)
