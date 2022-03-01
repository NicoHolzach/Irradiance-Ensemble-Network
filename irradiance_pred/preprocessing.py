"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
preprocessing.py defines the relevant functions to preprocess
each datatype used to train the model. It also defines a function
used to split the data into train (2014), validation (2015),
and test (2016) splits.
These splits can then be used to train, validate, and test models.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import os
import numpy as np
import pandas as pd
import cv2
from scipy import stats
import torch
from torch.utils.data import TensorDataset, DataLoader

# Define a function to preprocess our irradiance sequences.
def preprocess_irradiance(X, y):
	"""
	Function to preprocess irradiance data.
	This standardizes the X data for the model.
	It also brings the data into the right Pytorch format.

	Parameters
	----------
	X : np.array 
		Array containing irradiance sequences.
	y : np.array
		Arrray containing the targets.

	Returns
	-------
	(X, y) : tuple
		Tuple containing two preprocessed tensors (X, y).
	"""

	# Unstack, standardize, and stack X.
	num_rows = X.shape[0]
	out = X.flatten()
	out = stats.zscore(out)
	out = out.reshape((num_rows, -1))

	# Reshape X so it can be fed into our LSTM module.
	out = torch.from_numpy(out)
	out = out.view([len(out), out.shape[-1], -1])

	return (out.float(), torch.from_numpy(y).float())

# Define a function to preprocess our weather sequences.
def preprocess_weather(X, y, drop_precipation=False):
	"""
	Function to preprocess weather data.
	This standardizes each weather variable and also,
	if needed, drops the precipitation data due to 
	high number of zeros in data.
	It also brings the data into the right Pytorch format.

	Parameters
	----------
	X : np.array 
		Array containing weather sequences.
	y : np.array
		Arrray containing the targets.
	drop_precipation : bool (Default: True)
		Whether to drop precipitation column.

	Returns
	-------
	(X, y) : tuple
		Tuple containing two preprocessed tensors (X, y).
	"""

	# If specified, delete precipitation column.
	if drop_precipation == True:
		X = np.delete(X, 6, 2)
	
	# Unstack, standardize, and stack X.
	num_rows = X.shape[0]
	length = X.shape[1]
	X = np.reshape(X, (num_rows*length, -1))
	X = stats.zscore(X, axis=0)
	X = np.reshape(X, (num_rows, length, -1))

	return (torch.from_numpy(X).float(), torch.from_numpy(y).float())

# Define a function to preprocess our combined sequences.
def preprocess_combined(X, y):
	"""
	Function to preprocess irradiance+weather data.
	This standardizes each variable.
	It also brings the data into the right Pytorch format.

	Parameters
	----------
	X : np.array 
		Array containing combined sequences.
	y : np.array
		Arrray containing the targets.

	Returns
	-------
	(X, y) : tuple
		Tuple containing two preprocessed tensors (X, y).
	"""
	
	# Unstack, standardize, and stack X.
	num_rows = X.shape[0]
	length = X.shape[1]
	X = np.reshape(X, (num_rows*length, -1))
	X = stats.zscore(X, axis=0)
	X = np.reshape(X, (num_rows, length, -1))

	return (torch.from_numpy(X).float(), torch.from_numpy(y).float())

# Define a function to preprocess our static images.
def preprocess_images(X, y, resize=None):
	"""
	Function to preprocess skyimage data.
	This normalizes skyimage pixels per image. 
	If needed, it also resizes each image.
    Lastly, it brings X and y into the correct format for Pytorch.

	Parameters
	----------
	X : np.array 
		Array containing image data.
	y : np.array
		Arrray containing the targets.
	resize : (int, int) or None
			(Default: None)
		Tuple containing new (width, height) of image.

	Returns
	-------
	(X, y) : tuple
		Tuple containing two preprocessed tensors (X, y).
	"""

	# Either resize and normalize using max. pixel value of 255.
	if resize is not None:
		out = np.array([cv2.resize(img, (resize[0], resize[1]))
										.astype('float32') / 255.0 
						for img in X])
	
	# Or just normalize without resizing.
	else:
		out = np.array([img.astype('float32') / 255.0 
						for img in X])
		
	# Return in correct torch format.
	out = out.reshape(out.shape[0], 1, out.shape[1], out.shape[2])
	return (torch.from_numpy(out), torch.from_numpy(y).float())
    
# Define a function to preprocess our images3d sequences.
def preprocess_images3d(X, y, resize=None):
	"""
	Function to preprocess sequence of images3d data.
	This normalizes skyimage pixels per image. 
	If needed, it also resizes each image.
    Lastly, it brings X and y into the correct format for Pytorch.

	Parameters
	----------
	X : np.array 
		Array containing image3d sequences.
	y : np.array
		Arrray containing the targets.
	resize : (int, int) or None
			(Default: None)
		Tuple containing new (width, height) of image.

	Returns
	-------
	(X, y) : tuple
		Tuple containing two preprocessed tensors (X, y).
	"""
	
	# Either resize and normalize using max. pixel value of 255.
	if resize is not None:
		out = np.array([np.array([cv2.resize(img, (resize[0], resize[1]))
												.astype('float32') / 255.0 
                                	for img in img3d]) 
                            for img3d in X])
							
	# Or just normalize without resizing.
	else:
		out = np.array([np.array([img.astype('float32') / 255.0 
                                	for img in img3d]) 
                            for img3d in X])
		
	# Return in correct torch format
	out  = out.reshape(out.shape[0], 1, out.shape[1], out.shape[2], out.shape[3])
	return (torch.from_numpy(out), torch.from_numpy(y).float())

# Define a function that splits data into train, val, test sets
# and also applies the correct preprocessing.
def split_and_preprocess(X, y, timestamps, preprocessing, **kwargs):
	"""
    Function that splits the X and y data into 
    train, validation, and test splits. The splits are based on 
    the years 2014, 2015, and 2016 respectively. Each split is then
	preprocessed depending on their data type (image, irradiance, etc.)

    Parameters
	----------
    X : np.array
		Array containing model input data.
    y : np.array
		Array containing model targets.
    timestamps : np.array
		Array containing timestamps for each data point.
	preprocessing : {'irradiance', 'weather, 'combined', 'images', 'images3d', None}
		Which type of preprocessing to apply to the data.
	- **kwargs: other arguments depending on which data is preprocessed, i.e.:
		-- resize=(width, height) to resize images in skyimage data
		-- drop_precipation=bool to drop precipitation in weather data

    Returns
	-------
    X_train, X_val, X_test, y_train, y_val, y_test,
    timestamps_train, timestamps_val, timestamps_test : np.arrays
		Returns 9 numpy arrays, i.e. the train, val, and test data
		for the X, y, and timestamps data.  
    """

	# Define indices to filter each split.
	train_idx = [timestamp.year == 2014 for timestamp in timestamps]
	val_idx = [timestamp.year == 2015 for timestamp in timestamps]
	test_idx = [timestamp.year == 2016 for timestamp in timestamps]
	
	# Apply preprocessing depending on datatype.
	preprocess_dict = {'irradiance':preprocess_irradiance,
						'weather':preprocess_weather,
						'combined':preprocess_combined,
						'images':preprocess_images,
						'images3d':preprocess_images3d}
						
	if preprocessing is not None:
		X_train, y_train = preprocess_dict[preprocessing](X[train_idx], y[train_idx], **kwargs)
		X_val, y_val = preprocess_dict[preprocessing](X[val_idx], y[val_idx], **kwargs)
		X_test, y_test = preprocess_dict[preprocessing](X[test_idx], y[test_idx], **kwargs)
	else:
		X_train, y_train  = torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx])
		X_val, y_val  = torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx])
		X_test, y_test  = torch.from_numpy(X[test_idx]), torch.from_numpy(y[test_idx])
	
	# Create and return splits of X, y, and timestamps
    # in train, val, test order.
	return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            timestamps[train_idx], timestamps[val_idx], timestamps[test_idx])

# Define a function to create a Pytorch dataloader for mini-batch training.
def create_dataloader(X_train, X_val, y_train, y_val, batch_size, images_bool, GPU):
	"""
	Function that creates a Pytorch dataloader in order to
	train using mini-batches. Uses some of the output 
	from split_and_preprocess().

	Parameters
	----------
	X_train : torch.Tensor)
		Tensor containing training inputs.
	y_train : torch.Tensor
		Tensor containing training targets.
	X_val : torch.Tensor
		Tensor containing validation inputs.
	y_val : torch.Tensor
		Tensor containing validation targets.
	batch_size : int
		Size of mini batches for training.
	images_bool : bool
		Whether data are images or not and hence should be shuffled.
	GPU : bool
		Whether to train on the GPU (True) or on the CPU (False).

	Returns:
	(train_loader, val_loader) : torch.utils.data.DataLoader objects
		Tuple containing training and validation dataloaders.
	"""

	# Use GPU if specified.
	if GPU == True:

		if torch.cuda.is_available():
			X_train = X_train.cuda(0)
			y_train = y_train.cuda(0)
			X_val = X_val.cuda(0)
			y_val = y_val.cuda(0)
		
	# Combine X and y into datasets.
	train = TensorDataset(X_train, y_train)
	val = TensorDataset(X_val, y_val)

	# Create dataloaders with specified batch size and return them.
	train_loader = DataLoader(train, batch_size=batch_size, 
								shuffle=images_bool, drop_last=True)
	val_loader = DataLoader(val, batch_size=batch_size, 
								shuffle=images_bool, drop_last=True)

	return (train_loader, val_loader)


