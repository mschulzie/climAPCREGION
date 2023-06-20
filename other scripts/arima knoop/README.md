# climAPC_ARIMA
Contains functions to help fit a (S)ARIMA model to time series data, evaluate the model and to forecast/predict the timeseries. 

## Theory 
An ARIMA model consists of three components with their corresponding parameters:
  - AR(p): Autoregression
  - I(d): Integrated
  - MA(q): Moving average

AR and MA both make assumptions about the relation of the next timestep to the previous datapoints. In the AR model the order p indicates on how many previous values the current value $X_t$ depends: 
```math
X_t = \sum_{i=1}^{p}\psi X_{t-i} + \epsilon_t
```
with $\psi_{i-p}$ being the model parameters and $\epsilon_t$ as white noise.

If the raw observations are non-stationary, replacing the raw data by the difference $Z_t$ of their values to help with stationarity (which the other two models need). This is done d times:
```math
Z_t = X_{t+1} - X_t
```

The moving average model uses past forecast errors in a regression-like model, the order q dictates in this case the window size in which the errors are calculated:
```math
X_t = \mu \sum_{i=l}^{q}\theta_i  \epsilon_{t-i} + \epsilon_t
```
Where $\mu$ is the mean of the series, $\theta_{i-l}$ the model parameters and $\epsilon_{t-q}$ are the white noise error terms.

## How to choose (p,d,q)
A simple example of all the functions can be found in the example notebook.
The **examine_data()** function gives a good starting point for the first estimations of the orders of (p,d,q). The order of differencing I(d) should be determined first. If the data already seems stationary, no additional differencing has to be done (d=0), if the data looks like it has a linear trend, (d=1) is be needed. You can use different orders of d in examine_data(), to see which results in stationary data. Additional statistical test could also be [helpful](https://machinelearningmastery.com/time-series-data-stationary-python/).
The first guesses of q and p are a bit more tricky, but can be made with the halp of the Autocorrelation and Partial Autocorrelation. Help in these cases can be found for example [here](https://machinelearningmastery.com/time-series-data-stationary-python/) or [here](https://otexts.com/fpp2/non-seasonal-arima.html).   

## Constructing the first ARIMA model
The ARIMA model can be initialised and fitted with the **construct_ARIMA()** function, which makes use of [ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) from the statsmodels library. The result as well as the residuals and the ACF of the residuals are plotted. Under ideal circumstances teh residuals should be only white noise. To quickly evaluate the performance of a model the Akaike Information Critera (AIC) is printed, the lower the score, the better. Mny more evaluation tools for ARIMA models are available in the statsmodels toolbox.

## Prediction and Forecasting
One such method example would be fitting the model only to part of the data and comparing the prediction to the rest of the data this can be done with the **forecast_ARIMA()** function. In-sample predictions can be made by giving a **predict_index** in order to not start the forecast from the last value, but shifted forwards. With monthly data for example you would need a **predict_index = 12** to start one year earlier. This way the prediction can be compared to some real data and the root mean square error can be calculated. Standard forecasts into the future can be made by not setting a predict index (or setting it to 0).    

## Seasonal ARIMA
In order to better predict seasonal time series data, the SARIMA adds 4 new Parameters to an ARIMA model. The first three are (P,D,Q) but as seasonal parameters, the last one (m) is the number of datapoints per year. Finding (P,D,Q) is similar to finding the initial (p,d,q), more can be found [here](https://otexts.com/fpp2/seasonal-arima.html). 



### References 
Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting: principles and practice, 2nd edition, OTexts: Melbourne, Australia. OTexts.com/fpp2. Accessed on 15.06.2023.

[AR-model](https://en.wikipedia.org/wiki/Autoregressive_model)

[MA-model](https://en.wikipedia.org/wiki/Moving-average_model)