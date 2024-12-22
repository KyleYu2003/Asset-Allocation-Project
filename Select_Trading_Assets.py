#%%
from __future__ import print_function, absolute_import
from gm.api import *

with open("token.txt") as token_file:
    token = token_file.read()
set_token(token)

#%%
import pandas as pd
import numpy as np
from tqdm import tqdm

def string_to_timestamp(date_str):
    # Convert string to Timestamp with timezone 'Asia/Shanghai'
    timestamp = pd.Timestamp(date_str, tz='Asia/Shanghai')
    return timestamp

# read the file containing all ETFs in Shanghai Exchange.
ETFs = pd.read_excel("ETFs in Shanghai.xlsx")
ETFs_selected = ETFs[(ETFs.iloc[:, 1] == "单市场股票（沪）ETF") | (ETFs.iloc[:, 1] == "跨市场股票（沪深京）ETF")]
ETF_codes = ETFs_selected.iloc[:, 2]

data_entries = []
taded_amount = []

for code in tqdm(ETF_codes):
    # post adjust
    data = history(symbol='SHSE.' + str(code), frequency='1d', start_time='2022-09-01 09:00:00', end_time='2024-12-01 16:00:00',
               fields='amount,bob', adjust=ADJUST_POST, df=True)
    insample_data = data.loc[data.loc[:, "bob"] < string_to_timestamp('2024-09-01 09:00:00')]
    taded_amount.append(insample_data.loc[:, "amount"].sum())
    data_entries.append(data.shape[0])

ETFs_selected.loc[:, "data_entries"] = data_entries
ETFs_selected.loc[:, "taded_amount"] = taded_amount

ETFs_selected = ETFs_selected.sort_values(by=["data_entries", "taded_amount"], ascending=[False, False])
ETFs_selected.to_csv("sorted_ETFs.csv")

# %%
# The ETFs I selected are 512480, 512880, 512010, 515790, 512690, 512660, 512800, 512200, 516160, 512980, 515170, 516510
final_selected = [512480, 512880, 512010, 515790, 512690, 512660, 512800, 512200, 516160, 512980, 515170, 516510]
#%%
all_data = []

# download datas
for code in tqdm(final_selected):
    # post adjust
    # Assuming the slippage in trading could be neglected. Therefore, I directly use open to calculate return.
    data = history(symbol='SHSE.' + str(code), frequency='1d', start_time='2022-09-01 09:00:00', end_time='2024-12-01 16:00:00',
               fields='symbol,open,bob', adjust=ADJUST_POST, df=True)
    return_data = np.log(data.loc[:, "open"]/data.loc[:, "open"].shift(1))
    data.loc[:, "return"] = return_data
    data.drop("open", axis=1, inplace=True)
    data.loc[:, 'bob'] = pd.to_datetime(data.loc[:, 'bob']).dt.date
    all_data.append(data)

all_data = pd.concat(all_data, axis = 0)
all_data.to_csv("ETF_return_data.csv")