#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helpfuncs import difference,check_for_errors,fix_data  #functions from the helpfuncs.py script
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
#helpfull matplotlib parameters
props = dict(boxstyle='square', facecolor='lightgrey',edgecolor = 'red', alpha=0.9)#errorbox
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}

#%% methods

def examine_data(data,time, d = 0, warnings = True, set_nan = True):
    '''
    Generates plots of the Data (Timeseries, Autocorrelation function, Partial Autocorrelation function)
    Sets extreme outliers to NaN by default. Data can be temporarily differentiated for the plots, but won´t be returned as such

    Parameters
    ----------
    data : 1-D array-like
        Input timeseries.
    time : 1-D array-like
        Input time vector, same length as data.
    d : int, optional
        Temporary Order of differencing to examine possible higher orders of d. The default is 0.
    warnings : bool, optional
        prints possible warnings into the first plot. The default is True.

    Raises
    ------
    ValueError
        For not matching dimension between data and time.

    Returns
    -------
    NaN corrected data and timeseries, no differentiation

    '''
    #compare dimensions
    if data.shape != time.shape:
        raise ValueError(f"Timeseries data and time vector should have identical dimensions, not {data.shape} and {time.shape} ")
    
    #check for pandas series
    if isinstance(time,pd.Series):
        print('Pandas found; converting to numpy')
        time = time.to_numpy()
        data = data.to_numpy()
        
    #differentiate d times
    
    t_data,t_time = difference(data,time, d = d)
    
    #plot data,ACF and PACF
    fig = plt.figure(figsize=(6,10))

    ax = fig.add_subplot(3,1,1)
    ax.plot(t_time,t_data)
    ax.set_xlabel('Time')
    ax.set_title(f'Timeseries d={d} times differentiated')
    ax.grid()

    #check for errors 
    data,time,warn = check_for_errors(data, time, set_nan=set_nan)
    if warnings == False:
        warn = None
    
    #show errors on plot 1    
    ax.text(0.05, 0.95, warn, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
  
    #fix NaNs in order to correctly plot ACF and PACF
    data, time = fix_data(data, time, method = 'interp')
    
    #differentiate d times
    
    t_data,t_time = difference(data,time, d = d)
    
    #ACF
    ax = fig.add_subplot(3,1,2)#
    plot_acf(t_data, ax = ax)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.grid()
    
    #PACF
    ax = fig.add_subplot(3,1,3)
    plot_pacf(t_data, ax = ax)
    ax.set_ylabel('PACF')
    ax.set_xlabel('Lag')
    ax.grid()

    plt.tight_layout()
       
    return data,time



def construct_ARIMA(data,time,
                    order = (1,1,1),#AR(p),I(d),MA(q)
                    season = (0,0,0,0),# (Q,D,P,m)
                    manual = True,
                    plot_result = True,
                    ret_aic = False
                    ):
    '''
    

    Parameters
    ----------
    data : 1-D array-like
        Input timeseries.
    time : 1-D array-like
        Input time vector, same length as data.
    order : tuple (int,int,int) , optional
        (p,d,q) Parameters in the ARIMA model. p for autoregressive model AR(p);
        d for the order of integration; q for the moving-average model MA(q). Default is (1,1,1).
    season: bool or tuple:
       expands the model to a SARIMA model. must include the seasonal parameters (Q,D,P,m);
       m indicates the number of timesteps per year (e.g. 12 for monthly timeseries).
    manual : bool, optional
        wether to manually determine model parameters or let the statsmodels library. The default is True.
    plot_result : bool, optional
        generates figure of fitted data, residuals and the ACF of the residuals. The default is True.
    ret_aic : bool, optional
        additionally returns the Akaike information criterion to evaluate the model fit, lower is better. The default is True.

    Returns
    -------
    result : TYPE
        fitted ARIMA model object; can be parsed to the forecast function

    '''
    
    
    data,time,warn = check_for_errors(data, time, set_nan=False)
    if warn is not None:#fixing data if needed
        data,time = fix_data(data, time, method = "inter")
    
    #construct ARIMA model
    model = ARIMA(data, order = order,
                  seasonal_order = season,
                  missing = 'drop')
    
    result = model.fit()

    
    #plot best model?
    if plot_result:
        fig = plt.figure(figsize=(6,8))
    
        ax = fig.add_subplot(3,1,1)
        ax.plot(time,data, label = 'Data')
        ax.plot(time,result.fittedvalues, label = 'Fitted Model')
        ax.legend()
        
        ax = fig.add_subplot(3,1,2)
        ax.plot(time,result.resid)
        ax.set_title('Residuals')
        
        ax = fig.add_subplot(3,1,3)
        ax.plot(time,result.resid)
        ax.set_title('Autocorrelation of Residuals')
        ax.set_xlabel('Lags')
        ax.set_ylabel('ACF')
        
        plt.tight_layout()
    
    print(f'Fit complete: AIC for model of {order} and seasonality order {season}: {np.round(result.aic,2)} ')

    return result, data, time

def forecast_ARIMA(res,data,time, steps, predict_index = 0 ,
                   conf = 0.05,
                   plot_results = True,
                   ret_rmse = False):
    '''
    A Forecast function that uses the ARIMA model of the construct_ARIMA function and 
    forecasts from the last timestep into the future. In-sample predictions 
    from an earlier timestep are also possible (predict_index).
    
    To do an out-of-sample forecast of the next year with monthly timesteps:
            ->forecast_ARIMA(result, steps = 12, predict_index = 0)
    To do an in-sample prediction of the first 6 months of the last year : 
            -> forecast_ARIMA(result, steps = 6, predict_index = 12)
            
    Can also plot the results, and in case of an in-sample prediction calculate 
    the root mean square error (RMSE) towards the real timeseries.
    
    Parameters
    ----------
    res : Object wrapper
        result from the construct_ARIMA function; 
        ARIMAResultsWrapper object of statsmodels.tsa.arima.model module
    data : 1-D array-like
        Input timeseries.
    time : 1-D array-like
        Input time vector, same length as data.
    steps : int
        Number of timesteps that are to be predicted.
    predict_index : int, optional
        positive Index to run predictions . The default is 0.
    conf : float, optional
        ranges of the confidence interval. The default is 0.05 -> 5% and 95% 
    plot_results : bool, optional
        enables plotting. The default is True.
    ret_rmse : TYPE, optional
        if True returns the calculated RMSE as well . The default is False.


    Returns
    -------
    pred: array-like
        prediction with the dimensions of (1,steps).
    pred_time: array-like
        timevector for the predicted values.
    rmse:float
        RMSE; optional

    '''
    if predict_index <0:
        raise ValueError("The predict_index must be greater than or equal to 0.")
    
    rel_index = -predict_index #<=0
    
    #start and end points of Prediction/Forecast
    start = len(time) +1 + rel_index 
    end = start + steps -1
    #all +1 and -1 are there to counteract the negative index convention; so that 
    #a prediction of the last year has a rel_index of 12 in monthly data. 
    
    
    pred = res.predict(start, end, alpha = conf)#make prediction
    conf = res.get_prediction(start,end, alpha = conf).conf_int()#get confidence interv
    #generate timevector for prediction values
    dt = time[1] - time[0]
    
    
    if rel_index == 0:
        title = 'Forecast'
        print(title)
        pred_time = time[-1] + dt + np.arange(steps)*dt
        
        pred_time_plot = np.insert(pred_time, 0, time[-1])
        pred_plot = np.insert(pred, 0, data[-1])
        conf_plot = np.concatenate((np.array([[data[-1],data[-1]]]),conf))
    
    elif rel_index < 0:
        title = 'Prediction'
        print(title)
        #calc RMS error in case of overlapping data and predictions
        rms_data = data[start-1:end]
        rms_pred = pred[:-rel_index]
        rmse = np.mean(np.sqrt((rms_data - rms_pred)**2))/steps
        print(f'RMSE of prediction: {rmse}')
    
        #for better visuals connect first prediction with previous datapoint
        pred_time = time[start-1] + np.arange(steps)*dt
        pred_time_plot = np.insert(pred_time, 0, time[start-2])
        pred_plot = np.insert(pred, 0, data[start-2])
        conf_plot = np.concatenate((np.array([[data[start-2],data[start-2]]]),conf))
        
        
    if plot_results:
    
        fig = plt.figure(figsize=(6,4))
    
        ax = fig.add_subplot(1,1,1)
        ax.plot(time, data, label= 'data')
        ax.plot(pred_time_plot,pred_plot,lw = 3,ls = 'dashed', label = 'predict')
        
        ax.fill_between(pred_time_plot,#confidence interv
                    conf_plot[:, 0],
                    conf_plot[:, 1], color='k', alpha=.2)
        ax.set_xlim(time[start-steps*2],pred_time[-1]+2*dt)#appropriate zoom?
        ax.grid()
        ax.legend()
        ax.set_xlabel('Time')
        #title appropriate
        ax.set_title(title)
        if predict_index<steps:
            ax.set_title('Prediction/Forecast')
            
    if ret_rmse is not False:
        return pred,pred_time,rmse
    else:
        return pred, pred_time







