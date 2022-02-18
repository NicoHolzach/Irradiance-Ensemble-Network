# Irradiance-Ensemble-Network

This is the github repository containing all the necessary Python files and the structure of the directories for the analysis that trained and evaluated solar irradiance neural network ensemble forecasting models in the following paper: 

<p align="center">
    *CNN, LSTM, and Ensemble Neural Network  Solar Irradiance Forecasting:* <br />
    *Assessing the Impact  of Model Architecture, Input Type,  and Forecast Horizon*
</p>


The structure of this repository is as follows:

```
.
├── ...
├── irradiance_pred                   # Root directory containing all the files of the solar irradiance prediction analysis.
│   │                                  
│   ├── main.py                       # Python file that uses all the relevant functions from the other Python files
│   │                                   to conduct the analysis that is described in the paper.
│   │                                  
│   ├── data.py                       # Python file containing all the classes and functions that process the original data
│   │                                   to create the different data combinations used to train the models.
│   │                                  
│   ├── getperformance.py             # Python file containing functions that aggregate thes forecast errors of the models
│   │                                   and finds the best-performing parameters of the trained models.
│   │                                  
│   ├── models.py                     # Python file that defines the classes of the CNN, LSTM, and NN models. It also contains
│   │                                   the functions that define the different steps of the training process.
│   │                                  
│   ├── plotting.py                   # Python file containing all the functions used to plot the different graphs
│   │                                   that are used in the paper.
│   │                                  
│   ├── plotting.py                   # Python file containing the functions to pre-process the different data types
│   │                                   and to split them into train/val/test sets.
│   │                                  
│   ├── datasets                      # Directory containing all original and processed data used in this analysis.
│   │   │ 
│   │   ├── Folsom_irradiance.csv     # CSV file containing the original irradiance data (Pedro et al., 2019).
│   │   │ 
│   │   ├── Folsom_weather.csv        # CSV file containing the original weather data (Pedro et al., 2019).
│   │   │ 
│   │   ├── skyimages                 # Directory containing the original sky image data (Pedro et al., 2019).
│   │   │ 
│   │   ├── data_for_analysis         # Directory containing the data produced by data.py that are used to train the models.
│   │       │
│   │       ├── images                # Directory containing the input data (X) and their timestamps (t) for the images CNN model.
│   │       │
│   │       ├── images3d              # Directory containing the input data (X) and their timestamps (t) for the images3d CNN model.
│   │       │
│   │       ├── irradiance            # Directory containing the input data (X) and their timestamps (t) for the irradiance LSTM model.
│   │       │
│   │       ├── weather               # Directory containing the input data (X) and their timestamps (t) for the weather LSTM model.
│   │       │
│   │       ├── targets               # Directory containing the target data (y) and their timestamps to train and evaluate the models.
│   │       │
│   │       ├── predictions           # Directory containing the predictions of the individual models for all:
│   │                                   - data types (images, imaged3d, irradiance, weather, combined)
│   │                                   - data set splits (training, validation, testing)
│   │                                   - forecast horizons (15 min, 30 min, ..., 2 h, ..., 3 d)
│   │                                 
│   ├── parameters                    # Directory containing the trained model parameters for all 3069 models.
│   │   │                               Note that each of the following subdirectories contains a .pkl files that
│   │   │                               indexes each layer architecture for each model type.
│   │   │                               Also note that the parameters are further divided into subdirectories,
│   │   │                               which are not included in this repository for the sake of cleanliness.
│   │   │ 
│   │   │                               For example, imagine an LSTM irradiance model trained on:
│   │   │                               - delta=1, length=10, forecast horizon=30min
│   │   │                               - with a layer architecture indexed by 51
│   │   │ 
│   │   │                               Its trained model parameters would be stored in this subdirectory:
│   │   │                               ./parameters/irradiance/irradiance_layers_51/delta_1/length_10/y_30
│   │   │ 
│   │   ├── combined                  # Directory containing the trained model parameters of the combined LSTM model for all
│   │   │                               layer architectures, inputs, forecast horizons, and hyperparameter combinations.
│   │   │ 
│   │   ├── ensemble                  # Directory containing the trained model parameters of the ensemble NN model for all
│   │   │                               layer architectures, inputs, forecast horizons, and hyperparameter combinations.
│   │   │ 
│   │   ├── images                    # Directory containing the trained model parameters of the images CNN model for all
│   │   │                               layer architectures, inputs, forecast horizons, and hyperparameter combinations.
│   │   │ 
│   │   ├── images3d                  # Directory containing the trained model parameters of the images3d CNN model for all
│   │   │                               layer architectures, inputs, forecast horizons, and hyperparameter combinations.
│   │   │ 
│   │   ├── irradiance                # Directory containing the trained model parameters of the irradiance LSTM model for all
│   │   │                               layer architectures, inputs, forecast horizons, and hyperparameter combinations.
│   │   │ 
│   │   ├── weather                   # Directory containing the trained model parameters of the weather LSTM model for all
│   │                                   layer architectures, inputs, forecast horizons, and hyperparameter combinations.
│   │       
│   ├── performance                   # Directory containing the mini-batch training and validation RMSEs per epoch
│       │                               for each of the 3069 trained models.
│       │                               Note that this directory and its subdirectories follow the same structure as ./parameters.
│       │
│       │                               For example, imagine an LSTM irradiance model trained on:
│       │                               - delta=1, length=10, forecast horizon=30min
│       │                               - with a layer architecture indexed by 51
│       │
│       │                               Its mini-batch training and validatoin RMSEs would be stored in this subdirectory:
│       │                               ./performance/irradiance/irradiance_layers_51/delta_1/length_10/y_30
│       │
│       ├── combined                  # Directory containing the mini-batch RMSEs of the combined LSTM model for all
│       │                               layer architectures, inputs, forecast horizons, and hyperparameter combinations.
│       │
│       ├── ensemble                  # Directory containing the mini-batch RMSEs of the ensemble NN model for all
│       │                               layer architectures, inputs, forecast horizons, and hyperparameter combinations.
│       │
│       ├── images                    # Directory containing the mini-batch RMSEs of the images CNN model for all
│       │                               layer architectures, inputs, forecast horizons, and hyperparameter combinations.
│       │
│       ├── images3d                  # Directory containing the mini-batch RMSEs of the images3d CNN model for all
│       │                               layer architectures, inputs, forecast horizons, and hyperparameter combinations.
│       │
│       ├── irradiance                # Directory containing the mini-batch RMSEs of the irradiance LSTM model for all
│       │                               layer architectures, inputs, forecast horizons, and hyperparameter combinations.
│       │
│       ├── weather                   # Directory containing the mini-batch RMSEs of the weather LSTM model for all
│                                      layer architectures, inputs, forecast horizons, and hyperparameter combinations.
.
```

Please note that due to the file size of the data and model parameters, this repository only contains the Python files and the directories to convey the basic structure of the overall analysis. It does NOT include the actual data and model parameters. The directories in this repository DOES include paceholder files that mimick these original files.


# Source of the original irradiance, weather, and sky image data:

Pedro, H. T. C., Larson, D. P., and Coimbra, C. F. M. (2019). A comprehensive dataset
for the accelerated development and benchmarking of solar forecasting methods.
*Journal of Renewable and Sustainable Energy*, 11(3):036102. Publisher: American
Institute of Physics.

