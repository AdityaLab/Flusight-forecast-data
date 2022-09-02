"""
This is for parsing from CDC's github repo
    Output: score card containing all models
    i.e., 
    model, forecast_week, ahead, location (as region abbreviation), type, quantile, value
    e.g.
    GT-FluFNP, 202205, 1, CA, point, NaN, 843
    GT-FluFNP, 202205, 1, CA, quantile, 0.01, 338
        ....
        GT-FluFNP, 202205, 2, CA, point, NaN, 900
        GT-FluFNP, 202205, 2, CA, quantile, 0.01, 438
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
from epiweeks import Week
import pdb

# death_target = ['1 wk ahead inc death' , '2 wk ahead inc death' , '3 wk ahead inc death' , '4 wk ahead inc death']

data_ew = Week.thisweek(system="CDC") - 1  # -1 because we have data for the previous (ending) week
DIR =  './data-forecasts/'
models = [file.split(DIR)[1] for file in glob.glob(DIR + '/*') if ".md" not in file]
models.remove('GT-FluFNP-raw')
models.remove('Umass-ARIMA')
location_df = pd.read_csv('data-locations/locations.csv')
location_dict = {location_df['location'][i]:location_df['abbreviation'][i] for i in range(len(location_df))}
# for each model, get all submissions
df_list = []
model_point = ['CMU-TimeSeries', 'Flusight-ensemble', 'LUcompUncertLab-TEVA',
 'LUcompUncertLab-VAR2', 'LUcompUncertLab-VAR2K', 'LUcompUncertLab-VAR2K_plusCOVID', 'LUcompUncertLab-VAR2_plusCOVID',
 'LUcompUncertLab-humanjudgment','LosAlamos_NAU-CModel_Flu','UT_FluCast-Voltaire']
print(models)
for model in models:
    model_dir = DIR + '/' + model + '/' 

    all_items_path = np.array(glob.glob(model_dir + '*.csv'))  # list all csv files' paths
    all_items = [path.replace(model_dir, '') for path in all_items_path]  #list of all csv files' names

    """
    remove forecasts that were duplicated in a given week (if any)
    forecasts file should be unique for each epiweek
    """
    subm_dict = {}
    for i, item in enumerate(all_items):
        date = datetime.strptime(item[:10], '%Y-%m-%d')
        epiweek  = date.isocalendar()[1]
        if epiweek in subm_dict.keys():
            if subm_dict[epiweek][0] <= date:
                subm_dict[epiweek] = (date, i)
        else:
            subm_dict[epiweek] = (date, i)

    select = [ value[1] for key, value in subm_dict.items()]
    select_paths = all_items_path[select]


    data_model = []
    for path in select_paths:

        df = pd.read_csv(path)
        
        """
            create epiweek column
        """
        date = path.split('/')[-1][:10]
        # epiweek ends on Saturday, but submission is until Monday. 
        # we can subtract 2 days, thus, submission on Monday will be considered in the prev week  
        # this also aligns submission week and data
        date = datetime.strptime(date, '%Y-%m-%d') - timedelta(days=2)
        forecast_week = Week.fromdate(date)
        df['forecast_week'] = forecast_week
        #pdb.set_trace()
        data_model.append(df)


    # join all dataframes saved in data_model

    """
        select, rename and sort columns
    """
        

    """
        convert location to region abbreviation
    """
    print(model, 'predicted', len(data_model), 'weeks')
    df = pd.concat(data_model, ignore_index=True, sort=False)
    df = df.rename(columns={'target': 'ahead'})
    model_list = []
    df['location']= df['location'].astype(str)
    for i in range(len(df)):  
        key = df['location'][i]
        if len(key) == 1: 
            key = '0' + key
        df.at[i, 'location'] = location_dict[key]
        df.at[i, 'ahead'] = df['ahead'][i][0]
        model_list.append(model)
    df['model'] = model
    df = df[['model', 'forecast_week', 'ahead', 'location', 'type', 'quantile', 'value']]
    final_row = {'model': [], 'forecast_week': [], 'ahead':[], 'location':[],'type':[],'quantile':[],
             'value':[]}
    for index, row in df.iterrows():
        if row['quantile'] == 0.5 and model in model_point: 
            final_row['model'].append(row['model'])
            final_row['forecast_week'].append(row['forecast_week'])
            final_row['ahead'].append(row['ahead'])
            final_row['location'].append(row['location'])
            final_row['type'].append('point')
            final_row['quantile'].append(np.nan)
            final_row['value'].append(row['value'])
    df2 = pd.DataFrame(final_row)
    df3 = pd.concat([df,df2], ignore_index = False)
    df3 = df3.sort_values(by=['forecast_week', 'location', 'ahead', 'type'], ascending=[True, True,True,True])
    df_list.append(df3)  
df = pd.concat(df_list, ignore_index=True, sort=False)
df = df.sort_values(by=['model','forecast_week', 'location', 'ahead', 'type'], ascending=[True,True, True,True,True])
df.to_csv('./predictions.csv',index=False)
print("done")