# Irradiance-Ensemble-Network

This is the github repository containing all the necessary Python files and the basic structure of the data directories of the analysis that trained and evaluated a solar irradiance neural network ensemble forecasting model in the following paper: 

*CNN, LSTM, and Ensemble Neural Network  Solar Irradiance Forecasting: 
Assessing the Impact  of Model Architecture, Input Type,  and Forecast Horizon*

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
│   │   │   │
│   │   │   ├── images                # Directory containing the input data and their timestamps for the *images* CNN model.

└── ...
```


# Source of the original irradiance, weather, and sky image data:

Pedro, H. T. C., Larson, D. P., and Coimbra, C. F. M. (2019). A comprehensive dataset
for the accelerated development and benchmarking of solar forecasting methods.
*Journal of Renewable and Sustainable Energy*, 11(3):036102. Publisher: American
Institute of Physics.

