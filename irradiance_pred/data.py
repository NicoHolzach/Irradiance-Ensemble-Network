"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data.py defines the relevant functions to read in the data
and the relevant class that contains the data.
This class can then be used to generate samples to train
the models with.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


# Import general python libraries.
import itertools
import os
import gc
import pandas as pd
import numpy as np
import cv2
import datetime
import pytz
import random
import pickle5 as pickle
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Import our own modules.
import getperformance
import preprocessing

# Define the class used to manage the different datasets.
class DataSet():
	"""
	Class of all datasets used for the models.
	Contains the relevant data and methods to extract samples 
	of the X and y data. Also contains methods to extract the
	target y variable.
	"""
	# initialize class using respective pandas dataframe
	def __init__(self, df, type):
		"""
		Initializes the DataSet object.

		Parameters
		-----------
		df : pd.DataFrame 
			pandas dataframe to be stored in self.data
		type : {'images', 'weather', 'irradiance', 'combined'}
			Which data type to read in.

		Returns
		-------
		None
			Only initializes the DataSet class.

		"""
		self.data = df #dataframe containing aligned data and targets
		self.type = type #indicates if it includes only e.g. weather data
		self.sample = {} #empty dict that can be filled with 
						#different samples of timestamps
		
	# Create forecast targets.
	def create_targets(self, forecast_horizon):
		"""
		Class method that creates and saves the target values 
		for the prediction task in a dedicated .npy file.

		Parameters
		-----------
		forecast_horizon : int 
			How many minutes away the forecast should be
			e.g. 60 for an 1-hour-ahead forecast

		Returns
		-------
		num_targets (int) : 
			Returns number of targets created with this method.
			Also saves the targets and corresponding time stamps as .npy files.
		"""

		# Create name of desired target and get list of already created targets.
		col_name = f'y_{forecast_horizon}'
		existing_targets = os.listdir('./datasets/data_for_analysis/targets')

		# Terminate if targets were already created.
		if f'{col_name}.npy' in existing_targets:
			print(f'{forecast_horizon}-minutes-ahead targets already exist in datasets/data_for_analysis/targets')
			return None
        
        # Copy irradiance data into a new column
		# then shift this column up by the indicated number of minutes
		self.data[col_name] = self.data['ghi']
		self.data[col_name] = self.data[col_name].shift(-forecast_horizon)

		# save the targets and timestamps as numpy arrays
		self.save_data(
			full=False,
			datatype='targets',
			horizon=forecast_horizon)

		# drop the targets from the dataframe and return the number of targets created
		num_targets = len(self.data[self.data[col_name].notnull()][col_name])
		self.data.drop(labels=col_name, axis=1, inplace=True)
		return num_targets

	# Create skyimage sequences of defined frequencies and length.
	def create_images3d(self, delta, length):
		"""
		Class method to create sequences of historic skyimage data and to
		save them in a separate .npy file used for the 3D CNN model.

		Parameters
		-----------
		delta : int 
			Time interval between historic skyimages, 
			e.g. 10 for {length} skyimages with 10min intervals.
		length : int
			How many instances of historic skyimages should
			be collected, e.g. 3 to create a trio of skyimages
		
		Returns
		-------
		seq_shape : tuple
			Returns shape of sequence created with this method.
			Also saves the sequence and corresponding time stamps as .npy files.
		"""

		# Create name of desired sequence and get list of already created sequences.
		col_name = f'images3d_{delta}_{length}'
		existing_images3d = os.listdir('./datasets/data_for_analysis/images3d')

		# Terminate if sequence was already created.
		if f'{col_name}.npy' in existing_images3d:
			print(f'Skyimage sequence {delta} minutes apart and of length {length } already exists in this dataset.')
			return None

		# Add column for respective sequence to df to be filled later.
		self.data[col_name] = np.nan
		self.data[col_name] = self.data[col_name].astype('object')

		# Filter dataframe to only check timestamps where an image exists.
		index = self.data[self.data['images'].notnull()].index

		# Iterate through each timestamp and search for the skyimages.
		for i, idx in enumerate(index):
			
			# Get historic skyimages.
			try:
				images = [self.data.loc[idx \
								- datetime.timedelta(minutes=(length-j-1)*delta), \
									'images']
						for j in range(length)]
								
				# These images can either be numpy arrays or NaN.
				# So checking for float allows us to check if they're NaN.
				# If only one of them is NaN, we will exclude this sequence.
				if np.sum([isinstance(i, float) for i in images]) > 0:
					continue
					
				# If none are float, they are all arrays, hence they can be added to df.
				else:
					self.data.at[idx, col_name] = images

			# Skip if timestamp does not exist.
			except KeyError:
				continue
		
		# Save the skyimage sequence as numpy arrays.
		seq_shape = self.save_data(
			full=False,
			datatype='images3d',
			delta=delta,
			length=length,
			return_shape=True)
		
		# Drop the sequence from the dataframe and return the shape of the sequence.
		self.data.drop(labels=col_name, axis=1, inplace=True)
		gc.collect() #clear memory
		return seq_shape

	# Create sequence of historic irradiance for each timestamp.
	def create_irradiance(self, delta, length):
		"""
		Class method to create sequences of historic irradiance data and to
		save them in a separate .npy file used for the LSTM model.

		Parameters
		-----------
		delta : int
			Time interval between data points in sequence,
			e.g. 10 for {length} datapoints with 10min intervals.
		length : int)
			How many instances of historic irradiance should
			be collected, e.g. 20 for 20 instances of previous irradiance data
		
		Returns
		-------
		seq_shape : tuple
			Returns shape of sequence created with this method.
			Also saves the sequence and corresponding time stamps as .npy files.
		"""

		# Create the directory if needed.
		if not os.path.exists('./datasets/data_for_analysis/irradiance'):
			os.makedirs('./datasets/data_for_analysis/irradiance')
		
		# Create name of desired sequence and get list of already created sequences.
		col_name = f'irradiance_{delta}_{length}'
		existing_irradiance = os.listdir('./datasets/data_for_analysis/irradiance')

		# Terminate if sequence was already created.
		if f'{col_name}.npy' in existing_irradiance:
			print(f'Irradiance sequence {delta} minutes apart and of length {length } already exist in this dataset.')
			return None 

		# Create initial column that contains the current irradiance value.
		self.data['0'] = self.data['ghi']

		# Create additional columns that are shifted down by {delta} each.
		for i in range(1, length):
			self.data[str(delta*i)] = self.data['0'].shift(delta*i)

		# Aggregate all shifted columns into one large list.
		columns = [str(i*delta) for i in range(length)]
		columns = columns[::-1]
		self.data[col_name] = self.data[columns].values.tolist()

		# Drop the shifted columns from the df.
		self.data.drop(labels=columns, axis=1, inplace=True)
		gc.collect() # clean memory

		# Extract the irradiance sequence and timestamps 
		# but only consider timestamps with non-zero GHI.
		# Also prepare a boolean filter for them.
		irradiance = self.data[self.data['ghi'] != 0][col_name].to_numpy()
		timestamps = self.data[self.data['ghi'] != 0].index.to_numpy()
		nan_filter = [True for i in range(len(irradiance))]

		# Change the filter to exclude NaN.
		for i, v in enumerate(irradiance):
			
			# Ff its dtype is object, it will contain some NaN
			# hence, turn the filter at that instance to False
			# to filter out those dtype objects.
			if np.array(v).dtype == np.dtype('O'):
				nan_filter[i] = False
				continue
			
			# Some matrices might be of dtype=float64 but still contain NaN.
			# Thus, we must filter out those, too.
			if np.isnan(v).sum() > 0:
				nan_filter[i] = False

		# Apply the filter to the sequence and timestamps.
		nan_filter = np.array(nan_filter)
		irradiance = irradiance[nan_filter]
		irradiance = np.stack(irradiance)
		timestamps = timestamps[nan_filter]

		# Save the irradiance and timestamp data.
		np.save(file=f'./datasets/data_for_analysis/irradiance/{col_name}.npy',
				arr=irradiance)
		np.save(file=f'./datasets/data_for_analysis/irradiance/{col_name}_time.npy',
				arr=timestamps)

		# Get the shape of the sequence and clean the dataframe and memory.
		seq_shape = irradiance.shape
		self.data.drop(labels=col_name, axis=1, inplace=True)
		del irradiance
		del timestamps
		gc.collect()

		return seq_shape

	# Create sequence of historic weather data for each timestamp.
	def create_weather(self, delta, length):
		"""
		Class method to create sequences of historic weather data and to 
		save them as a separate .npy file used for the LSTM model.

		Parameters
		----------
		delta : int
			Time interval between data points in sequence, 
			e.g. 10 for {length} datapoints with 10min intervals
		length : int 
			How many instances of historic weather should
			be collected, e.g. 20 for 20 instances of previous irradiance data
		
		Returns
		-------
		seq_shape : tuple 
			Returns shape of sequence created with this method.
			Also saves the sequence and corresponding time stamps as .npy files.
		"""
		
		# Create the directory if needed.
		if not os.path.exists('./datasets/data_for_analysis/weather'):
			os.makedirs('./datasets/data_for_analysis/weather')
		
		# Create name of desired sequence and get list of already created sequences.
		col_name = f'weather_{delta}_{length}'
		existing_weather = os.listdir('./datasets/data_for_analysis/weather')

		# Terminate if sequence was already created.
		if f'{col_name}.npy' in existing_weather:
			print(f'Weather sequence {delta} minutes apart and of length {length } already exists in this dataset.')
			return None
		
		# Create initial column that aggregates all weather data into a list.
		self.data['0'] = self.data[['air_temp','relhum', 'press', \
								    'windsp', 'winddir', 'max_windsp', \
									'precipitation']].values.tolist()

		# Create additional columns that are shifted down by {delta} each.
		for i in range(1, length):
			self.data[str(delta*i)] = self.data['0'].shift(delta*i)
			
		# Aggregate all shifted columns into one large list.
		columns = [str(i*delta) for i in range(length)]
		columns = columns[::-1]
		self.data[col_name] = self.data[columns].values.tolist()

		# Drop the shifted columns from the df.
		self.data.drop(labels=columns, axis=1, inplace=True)
		gc.collect() # clean memory
		
		# Extract the weather sequence and timestamps
		# but only consider timestamps with non-zero GHI.
		# Also prepare a boolean filter for them.
		weather = self.data[self.data['ghi'] != 0][col_name].to_numpy()
		timestamps = self.data[self.data['ghi'] != 0].index.to_numpy()
		nan_filter = [True for i in range(len(weather))]
		
		# Change the filter to exclude NaN.
		for i, v in enumerate(weather):
			
			# If its dtype is object, it will contain some NaN
			# hence, turn the filter at that instance to False
			# to filter out those dtype objects.
			if np.array(v).dtype == np.dtype('O'):
				nan_filter[i] = False
				continue
			
			# Some matrices might be of dtype=float64 but still contain NaN
			# thus, we must filter out those, too.
			if np.isnan(v).sum() > 0:
				nan_filter[i] = False

		# Apply the filter to the sequence and timestamps.
		nan_filter = np.array(nan_filter)
		weather = weather[nan_filter]
		weather = np.stack(weather)
		timestamps = timestamps[nan_filter]

		# Save the weather and timestamp data.
		np.save(file=f'./datasets/data_for_analysis/weather/{col_name}.npy',
				arr=weather)
		np.save(file=f'./datasets/data_for_analysis/weather/{col_name}_time.npy',
				arr=timestamps)

		# Get the shape of the sequence and clean the dataframe and memory.
		seq_shape = weather.shape
		self.data.drop(labels=col_name, axis=1, inplace=True)
		del weather
		del timestamps
		gc.collect()

		return seq_shape

	# Create sample of timestamps that are shared across all sequences of this datatype.
	def create_sample(self, size, seed):
		"""
		Class method to create a random sample of timestamps of
		a specified sample size.

		Parameters
		----------
		size : float or str
			Size of sample (float), e.g. 0.1 for 10% of the data
			or frequency of timestamps (str), e.g., '3min' for 3min
			intervals of the data.
		seed : int
			Setting the seed of the random sample.

		Returns
		-------
		None
			Only adds np.array of sampled timestamps to self.sample.
		"""
		
		# Raise errors if size parameter has been wrongly input.
		if not isinstance(size, float) and not isinstance(size, str):
			raise TypeError('size must be either of type float or string.')
		
		# Define which folder to search for timestamps.
		folder_dict = {'images':'images3d',
                		'weather':'weather',
                		'irradiance':'irradiance',
						'combined':('irradiance', 'weather')}
		
		# If datatype is 'combined' we need to read the X data
		# in both the irradiance and weather folders.
		if self.type == 'combined':
			irr_path = f'./datasets/data_for_analysis/irradiance'
			weath_path = f'./datasets/data_for_analysis/weather'
			irr_X = [f'{irr_path}/{f}' for f in os.listdir(irr_path) \
										if 'time.npy' in f]
			weath_X = [f'{weath_path}/{f}' for f in os.listdir(weath_path) \
										if 'time.npy' in f]
			X_files = irr_X + weath_X
		
		# For the other datatypes, we can use their respective folders
		# to read in the X data.
		else:
			X_path = f'./datasets/data_for_analysis/{folder_dict[self.type]}'
			X_files = [f'{X_path}/{f}' for f in os.listdir(X_path) \
										if 'time.npy' in f]
			
		# Also read in the y data and combine all timestamps.
		y_path = f'./datasets/data_for_analysis/targets'
		y_files = [f'{y_path}/{f}' for f in os.listdir(y_path) \
										if 'time.npy' in f]
		timestamps = [pd.to_datetime(np.load(f)) for f in X_files + y_files]
		
		# Need to find the common timestamps across all timestamp files.
		# Building intersections across all timestamps sets takes too long,
		# hence, let's first create a dataframe for the first timestamps.
		time_df = pd.DataFrame(data=[0 for i in range(len(timestamps[0]))],
								index=timestamps[0],
								columns=['0'])
		
		# Now iteratively create a df for the other timestamps and merge them.
		for i, t in enumerate(timestamps[1:]):
			time_df2 = pd.DataFrame(data=[0 for i in range(len(t))],
                            		index=t,
                            		columns=[str(i)])
			time_df = time_df.merge(time_df2,
                            		left_index=True,
									right_index=True,
									how='inner')
									
		# The resulting merged dataframe will have the intersection
		# of all timestamps as an index.
		intersection = time_df.index
		
		# If the size parameter is a float,
		# this means that we want to create a sample of fixed length.
		if isinstance(size, float):
			intersection = list(intersection)
			sample_size = int(len(intersection) * size)
			random.seed(seed)
			sample = random.sample(intersection, sample_size)

			# Add this sample of timestamps to the sample dictionary.
			self.sample[size] = np.array(sample)

		# If the size parameter is a string
		# this means that we want to create a sample with a certain time frequency.
		elif isinstance(size, str):
			date_range = pd.date_range(
				start=intersection.min(),
				end=intersection.max(),
				freq=size)
			date_filter = np.in1d(intersection, date_range)

			# Add the sampled timestamps to the sample dict.
			self.sample[size] = np.array(intersection[date_filter])

	# Yield full irradiance (ghi) data
	def get_irradiance(self):
		return self.data['ghi']

	# Yield full weather data.
	def get_weather(self):
		return self.data[['air_temp', 'relhum', 'press', 'windsp',
						'winddir', 'max_windsp', 'precipitation']]
		
	# Return X, y, and time data for training, testing, validation.
	def get_data(self, datatype, forecast_horizon, sample, sample_size, delta, length):
		"""
        Class method that returns the X, y, and time data for modeling.
        Choose between which datatype, sample method,
		and what forecasting period. This is the main method used
		to extract the correctly aligned data for training,
		validating, and testing.
        
        Parameters
		----------
		datatype : {'images', 'images3d', 'irradiance', 'weather', 'combined'}
			Which type of data to fetch from the data_for_analysis folder.
			If 'combined', merges irradiance and weather sequences.
		forecast_horizon : int 
			How many minutes ahead the target y should be.
		sample : bool
			Whether to get whole data or just a sample of it.
		sample_size : float or str
			- If float, gets the sample of the data according to a fixed fraction,
			of the dataset size, e.g. 0.1 of all data
			- If str, gets the sample according to a certain frequency between
			timestamps, e.g. '3min' for 3 minute deltas.
			See https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
			for all accepted frequencies.
		delta : int 
			Indicates the distance (in minutes) between X data points.
			Only relevant for datatype = 'weather', 'irradiance', 'combined', or 'images3d'.
		length : int
			Indicates the length of historical X data.
			Only relevant for datatype = 'weather', 'irradiance', 'combined', or 'images3d'.
		
		Returns
		-------
		X : np.array
			X input for the model.
		y : np.array
			Target variable y for the model.
		timestamps : np.array
			Timestamps of the X and y data. 
			Can be used to split X and y into different sets based on time.
		"""
		
		# Choose the right data depending on the specified datatype
		X_dict = {'images':'images',
					'images3d':f'images3d_{delta}_{length}',
					'irradiance':f'irradiance_{delta}_{length}',
					'weather':f'weather_{delta}_{length}',
					'combined':(f'irradiance_{delta}_{length}',
								f'weather_{delta}_{length}')}
		X_name = X_dict[datatype]
		
		# Retrive X, y, and the timestamps for the specified datatype, input, and horizon.
		# Datatype 'combined' requires the X and time data to be merged.
		if datatype == 'combined':

			# Load in X and time data for irradiance and weather data
			irr_X = np.load(f'./datasets/data_for_analysis/irradiance/{X_name[0]}.npy')
			weath_X = np.load(f'./datasets/data_for_analysis/weather/{X_name[1]}.npy')
			irr_time = np.load(f'./datasets/data_for_analysis/irradiance/{X_name[0]}_time.npy')
			weath_time = np.load(f'./datasets/data_for_analysis/weather/{X_name[1]}_time.npy')
			irr_time = pd.to_datetime(irr_time)
			weath_time = pd.to_datetime(weath_time)

			# Find intersection of timestamps of both datatypes.
			combined_intersection = list((set(irr_time) & set(weath_time)))

			# We now need to create a filter for both X data based on this intersection.
			# For this, create a dataframe for each time array
			irr_df = pd.DataFrame(
				data=[False for i in range(len(irr_time))],
				index=irr_time,
				columns=['filter'])
			weath_df = pd.DataFrame(
				data=[False for i in range(len(weath_time))],
				index=weath_time,
				columns=['filter'])

			# Now turn each False into True for every date in the intersection.
			irr_df.loc[combined_intersection, 'filter'] = [True for i in range(len(combined_intersection))]
			weath_df.loc[combined_intersection, 'filter'] = [True for i in range(len(combined_intersection))]

			# We can now use this column for boolean filtering on our X data.
			irr_filter = irr_df['filter'].to_numpy()
			weath_filter = weath_df['filter'].to_numpy()
			irr_X = irr_X[irr_filter]
			weath_X = weath_X[weath_filter]

			# Now that both Xs are aligned by time, we can merge the arrays.
			# Also, we just need to filter one time array to get the X_time
			# as both filtered timestamps (irr or weather) will be identical.
			X = np.dstack((weath_X, irr_X))
			X_time = weath_time[weath_filter]
			
		# All the other datatypes can use their respective folders
		# to directly retrieve X and X_time.
		else:
			X = np.load(f'./datasets/data_for_analysis/{datatype}/{X_name}.npy')
			X_time = np.load(f'./datasets/data_for_analysis/{datatype}/{X_name}_time.npy')
			X_time = pd.to_datetime(X_time)
		
		# Also load in the target y data based on the specified horizon.
		y = np.load(f'./datasets/data_for_analysis/targets/y_{forecast_horizon}.npy')
		y_time = np.load(f'./datasets/data_for_analysis/targets/y_{forecast_horizon}_time.npy')
		y_time = pd.to_datetime(y_time)

		# Find which timestamps are represented in X and y data.
		intersection = list(set(X_time) & set(y_time))

		# We now need to create a filter for the X and y data based on this intersection.
		# For this, create a dataframe for each time array.
		X_df = pd.DataFrame(
			data=[False for i in range(len(X_time))],
			index=X_time,
			columns=['filter'])
		y_df = pd.DataFrame(
			data=[False for i in range(len(y_time))],
			index=y_time,
			columns=['filter'])

		# Now turn each False into True for every date in the intersection.
		X_df.loc[intersection, 'filter'] = [True for i in range(len(intersection))]
		y_df.loc[intersection, 'filter'] = [True for i in range(len(intersection))]

		# We can now use this column for boolean filtering on our data.
		X_filter = X_df['filter'].to_numpy()
		y_filter = y_df['filter'].to_numpy()
		X = X[X_filter]
		y = y[y_filter]
		timestamps = X_time[X_filter]

		# Get a sample of X, y, and t if needed.
		if sample == True:

			# For this, we use the dataframe method from above.
			# Filtering using a dataframe index like this is quite fast!
			sample_df = pd.DataFrame(
				data=[False for i in range(len(timestamps))],
				index=timestamps,
				columns=['filter'])

			# Get the correctly sampled timestamps depending on whether
			# the sample should be of a fixed size or with a specific 
			# frequency between timestamps.
			if isinstance(sample_size, float):
				sample_timestamps = self.sample[sample_size]
			elif isinstance(sample_size, str):
				sample_timestamps = self.sample[sample_size]
			else:
				raise TypeError('sample_freq must either be float or string.')

			# Turn each False into True for every date in the sample.
			sample_df.loc[sample_timestamps, 'filter'] = [True for i in range(len(sample_timestamps))]

			# Use this column to filter X, y, and timestamps based on the sample timestamps.
			sample_filter = sample_df['filter'].to_numpy()
			X = X[sample_filter]
			y = y[sample_filter]
			timestamps = timestamps[sample_filter]

		# Convert the timestamps from UTC to PST (California).
		pst = pytz.timezone('America/Los_Angeles')
		timestamps = timestamps.tz_localize(pytz.utc).tz_convert(pst)
		timestamps = timestamps.tz_localize(None)

		return(X, y, timestamps)
		
	# Save full data to pickle file.
	def save_data(self, full, datatype='targets', horizon=None, delta=None, length=None, return_shape=True):
		"""
		Class method to either save the full dataframe contained inin self.data 
		or just parts of it such as the sequence irradiance_5_10.

		Parameters
		----------
		full : bool
			Whether the full dataframe within self.data should be saved.
		datatype : {'targets', 'images', 'images3d', 'irradiance', 'weather'}
			Only needed if full=False.
			Picks which datatype should be saved. 
		horizon : int 
			Indicates the forecast horizon in minutes.
			Only needed if datatype='targets'
		delta : int
			Indicates the time (in min) between datapoints in sequence.
			Only needed if datatype is one of {'images3d', 'irradiance', 'weather'}.
		length int
			Indicates the length of the sequence to be saved.
			Only needed if datatype is one of {'images3d', 'irradiance', 'weather'}.
		return_shape : bool
			Whether to return the shape of the created sequence.

		Returns
		-------
		np.array.shape : tuple
			Returns the shape of the saved data (only if return_shape=True).
			Also saves .npy data to disc.
		"""

		# Either save the whole dataframe to a pickle file.
		if full == True:
			self.data.to_pickle(f'./datasets/{self.type}_full_data.pkl')
		
		# Or only save specific numpy arrays to a .npy file.
		elif full == False:
			
			# Choose the right data depending on the specified datatype.
			datatype_dict = {'images':'images',
							'targets':f'y_{horizon}',
							'images3d':f'images3d_{delta}_{length}',
							'irradiance':f'irradiance_{delta}_{length}',
							'weather':f'weather_{delta}_{length}'}
			col_name = datatype_dict[datatype]
			
			# Extract data and timestamps and shape them correctly.
			timestamps = self.data[self.data[col_name].notnull()].index.to_numpy()
			arr = self.data[self.data[col_name].notnull()][col_name].to_numpy()
			if datatype is not 'targets':
				arr = np.stack(arr)

			# Create the directory if needed.
			path = f'./datasets/data_for_analysis/{datatype}'
			dir = os.path.dirname(path)
			if not os.path.exists(dir):
				os.makedirs(dir)
			
			# Save data and timestamps in the corresponding folder.
			np.save(file=f'{path}/{col_name}.npy', \
					arr=arr)
			np.save(file=f'{path}/{col_name}_time.npy', \
					arr=timestamps)		
			print(f'Saved data and timestamps to {path}.')

			# Return the shape of the sequence data if needed.
			if return_shape == True:
				return arr.shape

# Define a function that reads in the irradiance data from the
# original source file. 
def read_irradiance_data():
	"""
	Function that reads in irradiance data from the original
	source .csv file.

	Parameters
	----------
	None

	Returns
	-------
	irradiance : pd.DataFrame
		Pandas dataframe with minute-by-minute irradiance data.
	"""

	# The original .csv file is missing a few dates, so we need
	# to create a date range to account for those.
	date_range = pd.date_range(start='2014-01-01-00:00:00',
							end='2017-01-01-00:00:00',
							freq='min')
	date_range.freq = None
	date_df = pd.DataFrame(index=date_range)
	date_df.index.name = 'timeStamp'

	# Read in csv file and generate pandas df.
	irradiance_path = './datasets/Folsom_irradiance.csv'
	irradiance = pd.read_csv(
		irradiance_path, 
		delimiter=',',
		index_col=0,
		parse_dates=True,
		header=0)

	# Merge the two dfs and return the result.
	irradiance = date_df.merge(
		irradiance,
		left_index=True,
		right_index=True,
		how='outer')

	return irradiance

# Define a function that reads in the weather data from the
# original source file. 
def read_weather_data():
	"""
	Function that reads in weather data from the original
	source .csv file.

	Parameters
	----------
	None

	Returns
	-------
	weather : pd.DataFrame
		Pandas dataframe with minute-by-minute weather data.
	"""

	# Read in csv file and generate pandas df.
	weather_path = './datasets/Folsom_weather.csv'
	weather = pd.read_csv(
		weather_path,
		delimiter=',',
		index_col=0,
		parse_dates=True,
		header=0)

	return weather


# Define a function that reads in the sky image data 
# either from original source files or from an already
# created pickle file containing all images.
def read_image_data(from_pkl, year_list=None, month_list=None):
	"""
	Function that reads in all 360 degree sky images from
	the respective folders sorted by year, month, and date
	containining the individual sky images

	Parameters
	----------
	from_pkl : bool
		Automatically fed in from the create_dataset() function.
		Indicates whether you want to create the data from scratch (False) 
		or if you want to read in a .pkl file that includes the merged image data
		and the already created forecast targets (True).
	year_list : list
		Array containing years to indicate which years of 
		skyimage data should be read, e.g. [2014, 2015]. 
		This is useful if the memory cannot handle all images at once.
	month_list : list)
		Array containing double-digit strings (e.g. '01')
		to indicate which months should be read, e.g. ['01', '02', '03'].
		This is useful if the memory cannot handle all images at once.

	Returns
	out or existing_data : pd.DataFrame
		Pandas dataframe where each row is one 3D matrix of image.
		This dataframe was either created from scratch (out)
		or created by reading in a picke file  (existing_data).
	"""

	# Initiliaze output.
	print('Reading sky images...')
	images_dict = {}

	# If from_pkl=False that means we want to read in all skyimages from all years. 
	# This would take a long time, hence we can split up the work. 
	# Thus, we can check if we have started this process,
	# i.e. if the file 'images_full_data.pkl' exist and which dates
	# are contained in that file.
	if from_pkl == False:
		try:
			existing_data = create_dataset(
				type='images',
				from_pkl=True)
			existing_data = existing_data.data #grab df from DataSet class
			existing_dates = existing_data.dropna(
				subset=['images'], 
				inplace=False)
			existing_years = existing_dates.index.year
			existing_months = existing_dates.index.month

			# Use these to create a list of year-month pairs of timestamps of existing images
			existing_dates = [f'{i[0]}-{i[1]}' for i in zip(existing_years, existing_months)]

			# However, sometimes an image "spills over" into the next month due to rounding of the date. 
			# To exclude these spillovers, we only consider months that contain >10 skyimages. 
			date_count = Counter(existing_dates)
			existing_dates = [i for i,v in date_count.items() if v > 10]
			
		except FileNotFoundError: #this means that we haven't started the process of reading all skyimages
			existing_data = None 
			existing_dates = []
			
	else: #this means that we do NOT want to read all skyimages from scratch
		existing_data = None
		existing_dates = []

	# Iterate through each year, month, and day
	# then read in each image of each minute of each day.
	if year_list == None:
		years = [2014, 2015, 2016]
	else:
		years = year_list
	
	for year in years:

		# Extract months and iterate through them.
		if month_list == None:
			months = [f for f in os.listdir(f'./datasets/skyimages/{year}') \
						if not f.startswith('.')]
		else:
			months = month_list

		for month in months:

			# We can now check if the respective year and month in the foor loop has
			# already been read in. If yes, we can skip reading in the images of that month.
			if f'{int(year)}-{int(month)}' in existing_dates:
				print(f'{year}-{month} already in "images_full_data.pkl"! Skipping to next month.')
				continue

			else:			
				# Extract days and iterate through them.
				days = [f for f in os.listdir(f'./datasets/skyimages/{year}/{month}') \
						if not f.startswith('.')]
				
				for day in days:
	
					image_list = [f for f in os.listdir(f'./datasets/skyimages/{year}/{month}/{day}') \
									if not f.startswith('.')]
								
					# Turn each image file into matrices, resize them, and grayscale them.
					for image in image_list:
						
						img = cv2.imread(f'./datasets/skyimages/{year}/{month}/{day}/{image}')
						img = cv2.resize(img, (200, 200))
						img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
						
						# Extract and round timestamp to nearest minute
						# then add everything to output dictionary.
						timestamp = roundTime(
							datetime.datetime(int(year), 
							int(month),
							int(day),
							int(image[9:11]),
							int(image[11:13]),
							int(image[13:15])))
						images_dict[timestamp] = [img]
						
				print(f'{year}-{month} done!')		

	# Create a dataframe from these skyimages.
	out =  pd.DataFrame.from_dict(
		images_dict,
		orient='index',
		columns=['images'])
	out.index.set_names('timeStamp', inplace=True)
	out['images'] = out['images'].apply(lambda x: np.array(x))

	# Ff we DON'T have or don't need to use previously read in skyimages, return df.
	if existing_data is None:
		return out

	# Ff we DO want to use previously read in skyimages, add the newly read in images
	# to the previous ones and then return the whole df.
	else:
		out.dropna(subset=['images'], inplace=True) #only keep dates with skyimages
		existing_data.loc[out.index, 'images'] = out['images']
		existing_data.drop('ghi', axis=1, inplace=True)

		return existing_data


# Define a function that rounds a timestamp to the nearest minute.
def roundTime(dt=None, roundTo=60):
   """
   Source: 
   https://stackoverflow.com/questions/3463930/how-to-round-the-minute-of-a-datetime-object/10854034#10854034
   
   Round a datetime object to any time lapse in seconds
   dt : datetime.datetime object, default now.
   roundTo : Closest number of seconds to round to, default 1 minute.
   Author: Thierry Husson 2012 - Use it as you want but don't blame me.
   """
   if dt == None : dt = datetime.datetime.now()
   seconds = (dt.replace(tzinfo=None) - dt.min).seconds
   rounding = (seconds+roundTo/2) // roundTo * roundTo
   return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)


# Define a function that creates an object using our DataSet class
# for the specified type of data by either reading in the already
# created dataframe or sourcing the data from scratch.
def create_dataset(type, from_pkl, **kwargs):
	"""
	Function that creates a DataSet class object for 
	the specified data type. It either reads in the data from
	an existing dataframe or reads them from scratch from the 
	original data source. 

	Parameters
	----------
	type : {'images', 'irradiance', 'weather', 'combined'}
		Which type of data this DataSet object should contain.
	from_pkl : bool 
		- If False, creates the data from scratch using the source files.
		- If True, reads in a .pkl file that includes the specified type of data
		and the already created forecast targets.
	**kwargs
		Other arguments that are fed into the respective functions
		to read in data, e.g. the 'year_list' argument in the read_image_data()
		function.

	Returns
	-------
	DataSet() : class object
		Returns a DataSet class object containing the selected type of data.
	"""
	# Either read data from existing .pkl file.
	if from_pkl == True:
		
		if type == 'combined':
			df = pd.read_pickle(f'./datasets/irradiance_full_data.pkl')
		elif type == 'images3d':
			df = pd.read_pickle(f'./datasets/images_full_data.pkl')
			return DataSet(df=df, type='images')
		else:
			df = pd.read_pickle(f'./datasets/{type}_full_data.pkl')
		
		return DataSet(df=df, type=type)

	# Irradiance data can be read with the read_irradiance_data() function.
	elif type == 'irradiance':
		df = read_irradiance_data()
		return DataSet(df=df, type='combined')

	# The combined data can also be read with the same function.
	elif type == 'combined':
		df = read_irradiance_data()
		return DataSet(df=df, type='combined')

	# The others need an additional ghi column used as the target for prediction.
	elif type == 'images' or type == 'images3d':
		ghi = read_irradiance_data()[['ghi']]
		df = read_image_data(from_pkl=from_pkl, **kwargs)
		df = ghi.merge(
			df,
			left_index=True,
			right_index=True,
			how='outer')
		return DataSet(df=df, type=type)

	elif type == 'weather':
		ghi = read_irradiance_data()[['ghi']]
		df = read_weather_data()
		df = ghi.merge(
			df,
			left_index=True,
			right_index=True,
			how='outer')
		return DataSet(df=df, type=type)

	else:
		raise ValueError('Must enter a valid value for the type parameter.')


# Define a function that creates a numpy dataset for the predictions
# made by the best model for a given datatype and horizon.
def create_prediction_data(datatype, horizons):
	"""
	Function that extracts and preprocesses the correct data for
	the given datatype and horizon. It then searches for the best
	model of this datatype-horizon combination and makes predictions
	on the train, test, and validation set. These predictions are then,
	together with the corresponding timestamps, saved in the
	data_for_analysis folder. These data can then be used for error analysis
	or to train ensemble models.

	Parameters
	----------
	datatype : {'images', 'images3d', 'irradiance', 'weather', 'combined'}
		Which datatype the model was trained on that should be used
		to make predictions.
	horizons : int or 1d-array
		- If int, makes predictions for this specified forecast horizon.
		- If 1d-array, makes predictions for each of the given horizons.

	Returns
	-------
	None
		Only saves .npy files in ./datasets/data_for_analysis/predictions.
	"""

	print(f'Starting the prediction process using {datatype} data to save the '
		'predictions for later analysis and training.')
	print('This might take a while...')
	
	# Change horizons to an array if needed.
	if isinstance(horizons, int):
		horizons = [horizons]

	# Create a dataset containing irradiance data.
	dataset = create_dataset(type='irradiance', from_pkl=True)

	# Iterate through each horizon and get the best model and the 
	# corresponding delta length of the input used to train the model.
	for horizon in horizons:
		model, delta, length = getperformance.get_best_model_by_horizon(
			datatype=datatype,
			horizon=horizon,
			return_delta_length=True, #get the best performing delta-length combination
			return_hyperparams=False,
			batch_size=None, #find the best model across all hyperparams
			lr=None,
			weight_decay=None)

		# Get the correct data depending on the datatype.   
		print(f'Extracting, preprocessing, and predicting on the {datatype} data '
			f'with delta={delta} and length={length} on a forecast horizon of {horizon} minutes')
		
		X, y, t = dataset.get_data(
			datatype=datatype,
			forecast_horizon=horizon,
			sample=False,
			sample_size=1,
			delta=delta,
			length=length)

		# Due to the size of the images and images3d datasets, preprocessing
		# the whole dataset can yield a RuntimeError due to memory constraints.
		# Thus, split the images and images3d datatset into 1000 chunks. 
		# Then preprocess each of these chunks and predict on each of them.
		# First, prepare the output.
		yhat_train = []
		yhat_val = []
		yhat_test = []
		t_train = []
		t_val = []
		t_test = []

		# This usage of chunks is mostly needed for the images/images3d datasets 
		# as the sequences for the LSTM are a lot smaller.
		if datatype in ['images', 'images3d']:
			num_chunks = 1000
		else:
			num_chunks = 25 #the lstm sequences can have smaller chunks

		for i in range(num_chunks):

			# Apply the correct preprocessing to each of the chunks.
			(X_train, X_val, X_test,
			_, _, _, #the true y's
			time_train, time_val, time_test) = preprocessing.split_and_preprocess(
				X=X[i::num_chunks], 
				y=y[i::num_chunks], 
				timestamps=t[i::num_chunks], 
				preprocessing=datatype)

			# Get predictions of the model for the train, val, and test set.
			# Then add each prediction and their timestamps to the output.
			yhat = model(X_train)
			yhat = yhat.detach().numpy().flatten()
			yhat_train.extend(yhat)
			t_train.extend(np.array(time_train))
			
			yhat = model(X_val)
			yhat = yhat.detach().numpy().flatten()
			yhat_val.extend(yhat)
			t_val.extend(np.array(time_val))

			yhat = model(X_test)
			yhat = yhat.detach().numpy().flatten()
			yhat_test.extend(yhat)
			t_test.extend(np.array(time_test))

		# Turn the predictions and timestamps into numpy arrays.
		yhat_train = np.array(yhat_train)
		yhat_val = np.array(yhat_val)
		yhat_test = np.array(yhat_test)
		t_train = np.array(t_train)
		t_val = np.array(t_val)
		t_test = np.array(t_test)

		# Get the indices of the sorted timestamp arrays.
		train_idx = np.argsort(t_train)
		val_idx = np.argsort(t_val)
		test_idx = np.argsort(t_test)

		# Use these indices to sort the arrays.
		yhat_train = yhat_train[train_idx]
		yhat_val = yhat_val[val_idx]
		yhat_test = yhat_test[test_idx]
		t_train = t_train[train_idx]
		t_val = t_val[val_idx]
		t_test = t_test[test_idx]

		# Create the directory if needed.
		prediction_dir = './datasets/data_for_analysis/predictions'
		if not os.path.exists(prediction_dir):
			os.makedirs(prediction_dir)
		
		# Save each prediction array.
		np.save(
			file=f'{prediction_dir}/predictions{horizon}_train_{datatype}.npy',
			arr=yhat_train)
		np.save(
			file=f'{prediction_dir}/predictions{horizon}_val_{datatype}.npy',
			arr=yhat_val)
		np.save(
			file=f'{prediction_dir}/predictions{horizon}_test_{datatype}.npy',
			arr=yhat_test)

		# Also save each timestamp.
		np.save(
			file=f'{prediction_dir}/predictions{horizon}_train_{datatype}_time.npy',
			arr=t_train)
		np.save(
			file=f'{prediction_dir}/predictions{horizon}_val_{datatype}_time.npy',
			arr=t_val)
		np.save(
			file=f'{prediction_dir}/predictions{horizon}_test_{datatype}_time.npy',
			arr=t_test)
		
		print('Successfully saved the predictions and their timestamps.')
		print('-------------------------------------')

	# Clean the large variables from memory.
	del X, X_train, X_val, X_test
	gc.collect()


# Define a function that yields the correct train and validation data
# used to train the ensemble models.
def get_ensemble_data(horizon, include_interactions=False, include_time=False, 
						include_hour_poly=False, include_ghi=False, include_weather=False,
						only_return_val_df=False, return_as_df=False):
	"""
	Function that searches the ./datasets/data_for_analysis/predictions
	directory and gets the outputs of the 5 models for the specified horizon.
	These predictions will be the X data for the ensemble model. It then
	searches for the correct y data in ./datasets/data_for_analysis/targets.
	These are then aligned based on the converted timestamps.

	Parameters
	----------
	horizon : int
		Gets the correct X and y data for this specified forecast horizon.
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
	only_return_val_df : bool (Default: False)
		If True, returns the whole dataframe containing the validation split data.
		Should be used to calculate the full validation RMSE but 
		should not be used for ensemble model training.
	return_as_df : bool (Default: False)
		- If True, returns the X_train, X_val, ... etc. data as a pandas.DataFrame
		w/o standardizing the values. (recommended for error analysis)
		- If False, returns the standardized data as numpy arrays (recommended for training)

	Returns
	-------
	X_train, X_val, X_test, 
	y_train, y_val, y_test, 
	t_train, t_val, t_test,
	labels : np.arrays/lists/pd.DataFrames
		Returns numpy arrays (or pd.DataFrames if return_as_df=True) containing input 
		and target data, and their timestamps, used to train, validate, and test the 
		ensemble models. Also returns labels for each input column, e.g. 
		'irradiance*weather' denoting the interaction term between the outputs 
		of the irradiance and weather models.
	"""

	# Initialize the output to store the X data.
	datatypes = ['images', 'images3d', 'irradiance', 'weather', 'combined']
	splits = ['train', 'val', 'test']
	X_dict = {datatype:{
				split:{'X':None, 'time':None} for split in splits}
				for datatype in datatypes}

	# Iterate through each datatype and split and get the correct X data
	# based on the specified forecast horizon.
	for datatype in datatypes:

		for split in splits:

			# Define the correct path, read in the X and timestamp data,
			# and save both in the correct place in the initialized dictionary.
			path = f'./datasets/data_for_analysis/predictions/predictions{horizon}_' \
				f'{split}_{datatype}'
			X = np.load(f'{path}.npy')
			t = np.load(f'{path}_time.npy')

			X_dict[datatype][split]['X'] = X
			X_dict[datatype][split]['time'] = t

	# We also need to load in the corresponding targets y, and their timestamps y_t.
	y_path = f'./datasets/data_for_analysis/targets/y_{horizon}'
	y = np.load(f'{y_path}.npy')
	y_t = np.load(f'{y_path}_time.npy')
	y_t = pd.to_datetime(y_t)

	# Note that the predictions of the models, i.e. the data in the X_dict, are already
	# converted to the PST timezone. Thus, we also need to convert the y targets to the
	# correct timezone.
	pst = pytz.timezone('America/Los_Angeles')
	y_t = y_t.tz_localize(pytz.utc).tz_convert(pst)
	y_t = y_t.tz_localize(None)

	# Now we need to align all data as not every timestamp has all 5 X datapoints and
	# the y datapoint. We could create sets out of each dataset and calculate their intersections
	# however, we can utilize pandas dataframes to achieve the same much quicker.
	# Let us first initialize the output variables where we'll store the aligned data.
	X_train, X_val, X_test, y_train, y_val, y_test, t_train, t_val, t_test = [None] * 9

	# Create a first dataframe from the y targets and their timestamps.
	y_df = pd.DataFrame(
		data=y,
		index=y_t,
		columns=['y'])

	# Now iterate through each datatype and data split. Create a dataframe for each
	# and merge it with the df containing the y data to align the timestamps and
	# to filter out the timestamps that are not in the intersection.
	for split in splits:

		aligned_df = y_df.copy() #this ensures that we can re-use the original df
								#across splits

		for datatype in datatypes:

			# Create a dataframe from the X and t data and merge it with the y dataframe.
			X_df = pd.DataFrame(
				data=X_dict[datatype][split]['X'],
				index=X_dict[datatype][split]['time'],
				columns=[datatype])
			aligned_df = aligned_df.merge(
				X_df,
				left_index=True,
				right_index=True,
				how='inner')

		# If needed, return the validation data df.
		if only_return_val_df:
			if split == 'val':
				return aligned_df

		# After all predictions of every datatype, i.e. X data for the ensemble models,
		# have been merged with the y data, we can extract the y and t data.
		y = aligned_df['y'].to_numpy()
		t = aligned_df.index.to_numpy()
		aligned_df.drop('y', inplace=True, axis=1)

		# We can now add other data, e.g. interactions terms, if specified.
		# First, let's add the interaction terms between all predictions.
		if include_interactions:
			
			# Iterate through all possible interaction terms and add each one to the df.
			for interaction in itertools.combinations(aligned_df.columns, 2):

				col_name = f'{interaction[0]}*{interaction[1]}'
				aligned_df[col_name] = aligned_df[interaction[0]] * aligned_df[interaction[1]]
		
		# Now add the month, day, and hour of the current prediction.
		if include_time:
			
			aligned_df['month'] = aligned_df.index.month
			aligned_df['day'] = aligned_df.index.day
			aligned_df['hour'] = aligned_df.index.hour

		# Add the polynomials of the hour to the data.
		if include_hour_poly:

			aligned_df['hour^2'] = aligned_df.index.hour ** 2
			aligned_df['hour^3'] = aligned_df.index.hour ** 3

		# Now add the current GHI to the data.
		if include_ghi:
			
			# Load in the irradiance data and convert the timezone to California.
			irradiance = create_dataset(type='irradiance', from_pkl=True)
			irradiance.data.index = irradiance.data.index.tz_localize(pytz.utc).tz_convert(pst)
			irradiance.data.index = irradiance.data.index.tz_localize(None)

			# Add the GHI data to the dataframe.
			aligned_df = aligned_df.merge(
				right=irradiance.data[['ghi']],
				how='inner',
				left_index=True,
				right_index=True)

		# Lastly, add the weather data if specified.
		if include_weather:
			
			# Load in the weather data and convert the timezone to California.
			weather = create_dataset(type='weather', from_pkl=True)
			weather.data.index = weather.data.index.tz_localize(pytz.utc).tz_convert(pst)
			weather.data.index = weather.data.index.tz_localize(None)
			weather.data.drop('ghi', axis=1, inplace=True)

			# Add the weather data to the dataframe.
			aligned_df = aligned_df.merge(
				right=weather.data,
				how='inner',
				left_index=True,
				right_index=True)

		# Standardize each column and extract X data and the labels.
		labels = list(aligned_df.columns)

		if return_as_df == False:
			X = StandardScaler().fit_transform(aligned_df)
		
		elif return_as_df == True: #we want the raw predictions for error analysis
			X = aligned_df

		# Save them to the corresponding output variable.
		if split == 'train':
			X_train, y_train, t_train = X, y, t
		elif split == 'val':
			X_val, y_val, t_val = X, y, t
		elif split == 'test':
			X_test, y_test, t_test = X, y, t

	
	return (X_train, X_val, X_test, y_train, y_val, y_test, t_train, t_val, t_test, labels)


# Define a function that aggregates summary statistics for all three
# major data types for each of the three years 2014-2016.
def summarize_data():
	"""
	Function that uses the the DataSet class to read in pickle
	files of dataframes with the image, irradiance, and weather data.
	It then aggregates summary statistics for each of these three
	types of data for each of the years 2014, 2015, and 2016.
	This aggregation of summary statistics is returned as a dataframe.

	Parameters
	----------
	None

	Returns
	-------
	summary_df : pd.DataFrame
		Pandas dataframe containing summary stastics aggregated by
		type of data and year of timestamp.
	"""

	# Create three data sets for the three major data types.
	images = create_dataset(type='images', from_pkl=True)
	irradiance = create_dataset(type='irradiance', from_pkl=True)
	weather = create_dataset(type='weather', from_pkl=True)

	# Convert the timestamps from UTC to PST (California).
	for dataset in [images, irradiance, weather]:

		df = dataset.data
		pst = pytz.timezone('America/Los_Angeles')
		df.index = df.index.tz_localize(pytz.utc).tz_convert(pst)
		df.index = df.index.tz_localize(None)

	# Drop the GHI columns from the images and weather dfs.
	images.data.drop('ghi', axis=1, inplace=True)
	weather.data.drop('ghi', axis=1, inplace=True)

	# Create the summary statistics of the irradiance and weather data.
	for dataset in [irradiance, weather]:

		# Group by year of the timestamp.
		df = dataset.data
		df = df.groupby(df.index.year).agg(
			{col:['min', 'max', 'mean', 'std', 'count']
			for col in df.columns})
		dataset.data = df

	# Create the summary statistics for the images.
	df_img = images.data
	df_img = df_img.groupby(df_img.index.year).agg(
		{'images':['count']})
	images.data = df_img

	# Add columns for the other stats to the images data.
	for stat in ['min', 'max', 'mean', 'std']:
		images.data.loc[:, ('images', stat)] = [np.nan for i in range(len(images.data))]

	# Only use years 2014-16 and then nstack the dataframes 
	# s.t. only the stats are columns.
	for dataset in [images, irradiance, weather]:

		# Drop 2013 if needed.
		df = dataset.data
		try:
			df = df.drop(2013)   
		except:
			KeyError

		# Create multiindex of year-variable pairs.
		df = df.unstack().unstack(level=1)
		df = df.reorder_levels([1, 0])
		df.sort_index(inplace=True)
		dataset.data = df

	# Reorder the columns of the images statistics.
	images.data = images.data[['min', 'max', 'mean', 'std', 'count']]

	# Concatenate all dataframes and return the whole df.
	df = pd.concat([images.data, irradiance.data, weather.data])
	df.sort_index(inplace=True, level=0)
	
	return df.round(2)

