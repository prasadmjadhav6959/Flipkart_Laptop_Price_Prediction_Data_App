import streamlit as st
import pandas as pd
import pickle
import os
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.subheader('Elon Musk Laptop Price Prediction')

FILE_DIR_1 = os.path.dirname(os.path.abspath(__file__))
FILE_DIR = os.path.join(FILE_DIR_1,os.pardir)
dir_of_interest = os.path.join(FILE_DIR, 'resourses')
DATA_PATH = os.path.join(dir_of_interest, 'data')

DATA_PATH_1 = os.path.join(DATA_PATH, 'laptop_price.csv')
df = pd.read_csv(DATA_PATH_1)
data = df.copy()
data.drop('Processor Type', axis=1, inplace=True)

col1, col2= st.columns(2)
with col1:
    brand = st.selectbox(
        'Select Laptop Brand',
        (df.Brand.unique()))

with col2:
    operating_system=st.selectbox(
        'Select Operating System',
        (df['Operating System'].unique()))

col1, col2= st.columns(2)
with col1:
    ram_type=st.selectbox(
        'Select RAM Type',
        (df['RAM Type'].unique()))

with col2:
    ram_size=st.selectbox(
        'Select RAM Size',
        (df['RAM Size'].unique()))

col1, col2= st.columns(2)
with col1:
    disc_type=st.selectbox(
        'Select DISC Type',
        (df['Disc Type'].unique()))

with col2:
    disc_size=st.selectbox(
        'Select DISC Size',
        (df['Disc Size'].unique()))

sample = pd.DataFrame({'Brand':[brand], 'Operating System':[operating_system],
                   'RAM Type':[ram_type], 'RAM Size':[ram_size],
                   'Disc Type':[disc_type], 'Disc Size':[disc_size]})

def replace_brand(brand):
    if brand == 'Lenovo':
        return 1
    elif brand == 'ASUS':
        return 2
    elif brand == 'HP':
        return 3
    elif brand == 'DELL':
        return 4
    elif brand == 'RedmiBook':
        return 5
    elif brand == 'realme':
        return 6
    elif brand == 'acer':
        return 7
    elif brand == 'MSI':
        return 8
    elif brand == 'APPLE':
        return 9
    elif brand == 'Infinix':
        return 10
    elif brand == 'SAMSUNG':
        return 11
    elif brand == 'Ultimus':
        return 12
    elif brand == 'Vaio':
        return 13
    elif brand == 'GIGABYTE':
        return 14
    elif brand == 'Nokia':
        return 15
    elif brand == 'ALIENWARE':
        return 16  
data['Brand'] = data['Brand'].apply(replace_brand)

def replace_os(os):
    if os == 'Windows 11':
        return 1
    elif os == 'Windows 10':
        return 2
    elif os == 'Mac':
        return 3
    elif os == 'Chrome':
        return 4
    elif os == 'DOS':
        return 5
data['Operating System'] = data['Operating System'].apply(replace_os)

def replace_ram_type(ram_type):
    if ram_type == 'DDR4':
        return 1
    elif ram_type == 'DDR5':
        return 2
    elif ram_type == 'LPDDR4':
        return 3
    elif ram_type == 'Unified':
        return 4
    elif ram_type == 'LPDDR4X':
        return 5
    elif ram_type == 'LPDDR5':
        return 6
    elif ram_type == 'LPDDR3':
        return 7   
data['RAM Type'] = data['RAM Type'].apply(replace_ram_type)

def replace_ram_size(ram_size):
    if ram_size == '8GB':
        return 1
    elif ram_size == '16GB':
        return 2
    elif ram_size == '4GB':
        return 3
    elif ram_size == '32GB':
        return 4
data['RAM Size'] = data['RAM Size'].apply(replace_ram_size)

def replace_disc_type(disc_type):
    if disc_type == 'SSD':
        return 1
    elif disc_type == 'HDD':
        return 2
    elif disc_type == 'EMMC':
        return 3
data['Disc Type'] = data['Disc Type'].apply(replace_disc_type)

def replace_disc_size(disc_size):
    if disc_size == '256GB':
        return 1
    elif disc_size == '512GB':
        return 2
    elif disc_size == '1TB':
        return 3
    elif disc_size == '128GB':
        return 4
    elif disc_size == '64GB':
        return 5
    elif disc_size == '32GB':
        return 6
    elif disc_size == '2TB':
        return 7
data['Disc Size'] = data['Disc Size'].apply(replace_disc_size)

X = data.drop('MRP', axis=1).values
y = data['MRP'].values

scaler = StandardScaler()
scaler_fit = scaler.fit(X)
X = scaler_fit.transform(X)

xg_reg = XGBRegressor(learning_rate=0.15, n_estimators=50, max_leaves=0, random_state=42)
xg_reg.fit(X, y)

sample['Brand'] = sample['Brand'].apply(replace_brand)
sample['Operating System'] = sample['Operating System'].apply(replace_os)   
sample['RAM Type'] = sample['RAM Type'].apply(replace_ram_type)
sample['RAM Size'] = sample['RAM Size'].apply(replace_ram_size)
sample['Disc Type'] = sample['Disc Type'].apply(replace_disc_type)
sample['Disc Size'] = sample['Disc Size'].apply(replace_disc_size)

sample = sample.values
sample = scaler_fit.transform(sample)

if st.button('Predict'):
    price = xg_reg.predict(sample)
    price = price[0].round(2)    
    st.text('Laptop Price: {}'.format('₹' + str(price)))
else:
    pass
