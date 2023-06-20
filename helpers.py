#%%
import pandas as pd
import xarray as xr

def get_TNH_idx(dropna=False):
    TNH = pd.read_table('data/tnh_index.tim',sep='\s+',header=5)
    info = pd.read_table('data/tnh_index.tim')[:4]
    TNH['time'] = [pd.Timestamp(year=year,month=month,day=1) for year, month in zip(TNH.YEAR,TNH.MONTH)]
    TNH = TNH.set_index('time').to_xarray().INDEX
    # drop months where pattern is not normally a leading mode of variability
    TNH = TNH.where(TNH!=-99.90) #  sets these values to nan
    if dropna:
        TNH = TNH.dropna('time')
    return TNH