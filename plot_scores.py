
# Ground truth is from covid-hospitalization-all-state-merged_vEW202210.csv

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
from epiweeks import Week
from metrics import *
EPS = 1e-6
import matplotlib.pyplot as plt


# In[2]:


# ground truth
df_ground_truth = pd.read_csv('ground_truth.csv') 


# In[3]:


df_ground_truth.head()
df_grnd = df_ground_truth[['epiweek', 'region', 'cdc_flu_hosp']]
df_grnd = df_grnd[df_grnd['epiweek']>=202201]
df_grnd = df_grnd.rename(columns = {'epiweek':"predicted_week", "cdc_flu_hosp":"value", "region":"location"})
df_grnd['location'] = df_grnd['location'].str.replace('X', 'US')
df_grnd['location'] = df_grnd['location'].str.replace('TUS', 'TX')
df_grnd = df_grnd.sort_values('location', kind = 'mergesort')
# df_grnd.head()


# In[4]:


file_dir = './predictions.csv' 
df_total = pd.read_csv(file_dir)


# In[5]:


df_total['model'].nunique()
df_final = df_total.copy()
all_model_names = np.array(df_final['model'].drop_duplicates())


# In[7]:


all_model_names = np.array(df_final['model'].drop_duplicates())
df_gt = df_final[df_final['model']=='GT-FluFNP']

# GT-FluFNP model hasn't predicted for some locations 
all_regions = np.array(df_gt['location'].drop_duplicates())
regions_ground_truth = np.array(df_grnd['location'].drop_duplicates())


# In[10]:


df_point = df_final[df_final['type']=='point']
df_quant = df_final[df_final['type']=='quantile']


# In[12]:


weeks = np.array(df_point['forecast_week'].drop_duplicates())
max_week = df_grnd['predicted_week'].max()


# In[14]:


df_point['predicted_week'] = df_point['forecast_week']+df_point['ahead']

# Have ground truth only till week 10 (max week)  
df_point = df_point[df_point['predicted_week']<=max_week] 


# In[15]:


# Merging the two datasets on predicted week
df_newpoint = pd.merge(df_point, df_grnd, on = "predicted_week")
# Removing all unnecessary merges
df_newpoint = df_newpoint[df_newpoint['location_x'] == df_newpoint['location_y']]


# In[16]:


rmse_all = []
model_all= []
mape_all = []
week_ahead = []
regions = []


# In[17]:


for model in all_model_names:
    for i in range(1, 5):
        for region in all_regions:
            sample = df_newpoint[   (df_newpoint['model']==model)  &   (df_newpoint['ahead']==i)  & (df_newpoint['location_x']==region) ]['value_x'].values
            target = df_newpoint[   (df_newpoint['model']==model)  &   (df_newpoint['ahead']==i)  & (df_newpoint['location_x']==region) ]['value_y'].values
            rmse_all.append(rmse(sample, target))

#             Deal with inf values
            target = np.array([EPS if x ==0 else x for x in target]).reshape((len(target), 1))
            mape_all.append(mape(sample, target))
            model_all.append(model)
            week_ahead.append(i)
            regions.append(region)


# In[18]:


df_point_scores = pd.DataFrame.from_dict({'Model':model_all, 'RMSE':rmse_all, 'MAPE':mape_all, 'Weeks ahead':week_ahead, 'Location':regions})


# In[20]:


df_point_scores.to_csv('point_scores.csv')


# In[21]:


# target is ground truth
df_quant = df_final[df_final['type']=='quantile']


# In[23]:


# norm_val = (df_quant['value']-df_quant['value'].min())/(df_quant['value'].max()-df_quant['value'].min())

norm_df_quant = df_quant.copy()
norm_df_quant['predicted_week']= norm_df_quant['forecast_week']+norm_df_quant['ahead']
norm_df_quant = norm_df_quant[norm_df_quant['predicted_week']<=max_week] 


# In[38]:


week_ahead = []
regions = []
crps_all = []
ls_all = []
model_all = []
cs_all = []


# In[39]:


# df_newpoint = df_newpoint[df_newpoint['model']!='GH-Flusight']
# norm_df_quant = norm_df_quant[norm_df_quant['model']!='GH-Flusight']


# Problem - some 'ahead' start differently?
# Problem - some weeks start from 202205
import warnings
warnings.filterwarnings("ignore")


# In[40]:


# All models
for model in all_model_names:
    print('Model ', model)
    
#     All Weeks ahead
    for i in range(1, 5):
        print('Week ahead ', i)
        
#         All regions
        for region in all_regions:
            
#             Dataset with information about Ground truth ('value_y') and predictions ('value_x') 
            target = df_newpoint[ (df_newpoint['model']==model) & (df_newpoint['ahead']==i) & (df_newpoint['location_x']==region)]
            
            norm_model = norm_df_quant[   (norm_df_quant['model']==model)  &   (norm_df_quant['ahead']==i)  & (norm_df_quant['location']==region) ]
            mean_ = []
            std_ = []
            var_ = []
            tg_vals = []
            pred_vals = []
            
            weeks = np.array(target['forecast_week'].drop_duplicates())
            if(len(weeks)!=0):
                for week in weeks:
    #                 Append point predictions
                    point_val = target[ (target['forecast_week']==week)]['value_x'].values
                    mean_.append(point_val)
                    if(len(point_val)==0):
                        print(i, week, region, model)

    #                 Append point pred as predictions
                    predval = target[ (target['forecast_week']==week)]['value_y'].values 
                    pred_vals.append(predval)
                
    #                     Append ground truth as target
                    tgval = target[ (target['forecast_week']==week)]['value_y'].values
                    tg_vals.append(tgval)

    #                 Find std from quantiles
                    b = norm_model[  (norm_model['forecast_week']==week) &  (norm_model['quantile']==0.75)]['value'].values
                    a = norm_model[  (norm_model['forecast_week']==week) &  (norm_model['quantile']==0.25)]['value'].values
                    std = (b-a)/1.35
                    var = std**2
                    std_.append(std)
                    var_.append(var)

                std_ = np.array(std_)
                var_ = np.array(var_)
#                 print('var - ', var_.shape, '\n')
                
                pred_vals = np.array(pred_vals)
#                 print('pred - ', pred_vals.shape, pred_vals, '\n')
                
                mean_ = np.array(mean_)
                tg_vals = np.array(tg_vals)
#                 print('tgvals - ', tg_vals.shape, tg_vals, '\n')


                if(len(tg_vals)==0):
                    print(i, region, model)
    #             Calculate ls and crps
                cr = crps(mean_, std_, tg_vals)
                auc, cs, _ = get_pr(pred_vals, var_, tg_vals)
                ls = log_score(mean_, std_, tg_vals, window = 0.1)
                if(ls<-10):
                    ls = -10
                elif(ls>10):
                    ls = 10
                crps_all.append(cr)
                ls_all.append(ls)
#                 print(cs)
                cs_all.append(cs)
                
            else:
                crps_all.append(np.nan)
                ls_all.append(np.nan)
                cs_all.append(np.nan)
            week_ahead.append(i)
            regions.append(region)
            model_all.append(model)
                


# In[41]:


len(regions)


# In[43]:


df_spread_scores = pd.DataFrame.from_dict({'Model':model_all, 'Weeks ahead':week_ahead, 'Location':regions, 'LS':ls_all, 'CRPS':crps_all,'CS':cs_all})


# In[44]:


df_spread_scores.head()
df_spread_scores[df_spread_scores['Model']=='GT-FluFNP']


# In[45]:


df_spread_scores.to_csv('spread_scores.csv')

