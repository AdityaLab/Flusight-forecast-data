# Ground truth is from covid-hospitalization-all-state-merged_vEW202210.csv

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
from epiweeks import Week
from metrics import *

EPS = 1e-6
import matplotlib.pyplot as plt
import math


# In[2]:


# ground truth
df_ground_truth = pd.read_csv("ground_truth.csv")


# In[3]:


df_ground_truth.head()
df_grnd = df_ground_truth[["epiweek", "region", "cdc_flu_hosp"]]
df_grnd = df_grnd[df_grnd["epiweek"] >= 202201]
df_grnd = df_grnd.rename(
    columns={"epiweek": "predicted_week", "cdc_flu_hosp": "value", "region": "location"}
)
df_grnd["location"] = df_grnd["location"].str.replace("X", "US")
df_grnd["location"] = df_grnd["location"].str.replace("TUS", "TX")
df_grnd = df_grnd.sort_values("location", kind="mergesort")
# df_grnd.head()


# In[4]:


file_dir = "./predictions.csv"
df_total = pd.read_csv(file_dir)


# In[5]:


df_total["model"].nunique()
df_final = df_total.copy()
all_model_names = np.array(df_final["model"].drop_duplicates())


# In[6]:


all_model_names = np.array(df_final["model"].drop_duplicates())
df_gt = df_final[df_final["model"] == "GT-FluFNP"]

# GT-FluFNP model hasn't predicted for some locations
all_regions = np.array(df_gt["location"].drop_duplicates())
regions_ground_truth = np.array(df_grnd["location"].drop_duplicates())


# In[7]:


df_point = df_final[df_final["type"] == "point"]
df_quant = df_final[df_final["type"] == "quantile"]


# In[8]:


weeks = np.array(df_point["forecast_week"].drop_duplicates())
max_week = df_grnd["predicted_week"].max()


# In[9]:


df_point["predicted_week"] = df_point["forecast_week"] + df_point["ahead"]

# Have ground truth only till week 10

df_point = df_point[df_point["predicted_week"] <= max_week]


# In[10]:


# Merging the two datasets on predicted week
df_newpoint = pd.merge(df_point, df_grnd, on="predicted_week")
# Removing all unnecessary merges
df_newpoint = df_newpoint[df_newpoint["location_x"] == df_newpoint["location_y"]]


# In[11]:


rmse_all = []
nrmse_all = []
model_all = []
mape_all = []
week_ahead = []
regions = []


# In[ ]:


for model in all_model_names:
    for i in range(1, 5):
        for region in all_regions:
            sample = df_newpoint[
                (df_newpoint["model"] == model)
                & (df_newpoint["ahead"] == i)
                & (df_newpoint["location_x"] == region)
            ]["value_x"].values
            target = df_newpoint[
                (df_newpoint["model"] == model)
                & (df_newpoint["ahead"] == i)
                & (df_newpoint["location_x"] == region)
            ]["value_y"].values
            rmse_all.append(rmse(sample, target))
            nrmse_all.append(norm_rmse(sample, target))

            #             Deal with inf values
            target = np.array([EPS if x == 0 else x for x in target]).reshape(
                (len(target), 1)
            )
            mape_all.append(mape(sample, target))
            model_all.append(model)
            week_ahead.append(i)
            regions.append(region)


# In[ ]:


df_point_scores = pd.DataFrame.from_dict(
    {
        "Model": model_all,
        "RMSE": rmse_all,
        "NRMSE": nrmse_all,
        "MAPE": mape_all,
        "Weeks ahead": week_ahead,
        "Location": regions,
    }
)


# In[ ]:


df_point_scores.to_csv("point_scores.csv")


# In[12]:


# target is ground truth
df_quant = df_final[df_final["type"] == "quantile"]


# In[13]:


# norm_val = (df_quant['value']-df_quant['value'].min())/(df_quant['value'].max()-df_quant['value'].min())

norm_df_quant = df_quant.copy()
norm_df_quant["predicted_week"] = (
    norm_df_quant["forecast_week"] + norm_df_quant["ahead"]
)
norm_df_quant = norm_df_quant[norm_df_quant["predicted_week"] <= max_week]


# In[64]:


week_ahead = []
regions = []
crps_all = []
ls_all = []
model_all = []
cs_all = []


# In[65]:


# Runtime warning - invalid value occurs during multiply -- ignore
import warnings

warnings.filterwarnings("ignore")


# In[66]:


# All models
count = 0
for model in all_model_names:
    print("Compiling scores of model ", model)
    print(f"Model {count}/{len(all_model_names)}")
    count += 1

    #     All Weeks ahead
    for i in range(1, 5):
        print("Week ahead ", i)

        #         All regions
        for region in all_regions:

            #             Dataset with information about Ground truth ('value_y') and predictions ('value_x')
            target = df_newpoint[
                (df_newpoint["model"] == model)
                & (df_newpoint["ahead"] == i)
                & (df_newpoint["location_x"] == region)
            ]

            norm_model = norm_df_quant[
                (norm_df_quant["model"] == model)
                & (norm_df_quant["ahead"] == i)
                & (norm_df_quant["location"] == region)
            ]
            mean_ = []
            std_ = []
            var_ = []
            tg_vals = []
            pred_vals = []

            weeks = np.array(target["forecast_week"].drop_duplicates())
            if len(weeks) != 0:
                for week in weeks:
                    #                 Append point predictions
                    point_val = target[(target["forecast_week"] == week)][
                        "value_x"
                    ].values
                    mean_.append(point_val)
                    if len(point_val) == 0:
                        print(i, week, region, model)

                    #                 Append point pred as predictions
                    predval = target[(target["forecast_week"] == week)][
                        "value_y"
                    ].values
                    pred_vals.append(predval)

                    #                     Append ground truth as target
                    tgval = target[(target["forecast_week"] == week)]["value_y"].values
                    tg_vals.append(tgval)

                    #                 Find std from quantiles
                    b = norm_model[
                        (norm_model["forecast_week"] == week)
                        & (norm_model["quantile"] == 0.75)
                    ]["value"].values
                    a = norm_model[
                        (norm_model["forecast_week"] == week)
                        & (norm_model["quantile"] == 0.25)
                    ]["value"].values
                    std = (b - a) / 1.35

                    var = std**2
                    std_.append(std)
                    var_.append(var)

                std_ = np.array(std_)
                var_ = np.array(var_)
                pred_vals = np.array(pred_vals)
                mean_ = np.array(mean_)
                tg_vals = np.array(tg_vals)

                if len(tg_vals) == 0:
                    print(
                        "No target found for week ahead ",
                        i,
                        " region ",
                        region,
                        "model ",
                        model,
                    )

                #
                #                     print(cr, ls)

                #             Calculate ls and crps
                cr = crps(mean_, std_, tg_vals)
                ls = log_score(mean_, std_, tg_vals, window = 0.1)
                if(ls<-10):
                    ls = -10
#                     print(cr, ls, "hi")
                auc, cs, _ = get_pr(mean_, std_**2, tg_vals)

                
#                 if(ls<-10 or math.isnan(ls)):
#                     ls = -10
#                 elif(ls>10):
#                     ls = 10
#                 if(math.isnan(cr)):
#                     cr = 0
                    
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


# In[67]:


df_spread_scores = pd.DataFrame.from_dict(
    {
        "Model": model_all,
        "Weeks ahead": week_ahead,
        "Location": regions,
        "LS": ls_all,
        "CRPS": crps_all,
        "CS": cs_all,
    }
)


# In[68]:


df_spread_scores.isna().sum()


# In[70]:


df_spread_scores.to_csv("spread_scores.csv")
