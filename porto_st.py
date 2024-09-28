import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import io

DATA_URL = 'data_kaggle.csv'

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    return data

data = load_data()

st.title('Property Listings in Kuala Lumpur')

st.header('--==Data Understanding==--')
st.subheader('Raw Dataset')
st.write(data.head())

st.subheader('Raw Dataset Information')
st.write('Description')
st.write(data.describe(include='all'))
st.write('------')
st.write('Number of uniques')
st.write('Unique Location :', data['Location'].nunique())
st.write('Unique Price :', data['Price'].nunique())
st.write('Unique Rooms :', data['Rooms'].nunique())
st.write('Unique Property Type :', data['Property Type'].nunique())
st.write('Unique Size :', data['Size'].nunique())
st.write('Unique Furnishing :', data['Furnishing'].nunique())
st.write('------')
st.write('Number of null values')
st.write(data.isna().sum())
st.write('------')
st.write('Number of duplicates')
st.write(data.duplicated().sum())

st.header('--==Data Preparation==--')
st.subheader('Sampling')
df_sample = data.sample(frac=0.2)
df_sample = df_sample.reset_index()
df_sample = df_sample.drop(['index'], axis=1)
st.write(df_sample.describe(include='all'))

st.subheader('Mengubah Price Menjadi Numeric')
df_price = df_sample.copy()
df_price = df_price['Price'].replace({'RM ': '', ',': ''}, regex=True).astype(float)
df_floatPrice = df_sample.copy()
df_floatPrice.Price = df_price
st.write(df_floatPrice.head())

st.subheader('Drop Kolom Size')
df_noSize = df_floatPrice.drop(['Size'], axis=1)
st.write(df_noSize.head())

st.subheader('Ringkas Location')
df_location = df_noSize['Location'].replace({', Kuala Lumpur': ''}, regex=True)
df_shortLoc = df_noSize.copy()
df_shortLoc.Location = df_location
st.write(df_shortLoc.head())

st.subheader('Mengubah rooms menjadi numeric')
df_rooms = df_shortLoc.copy()
df_rooms = df_rooms['Rooms'].replace({'4\+1': '5', '3\+1': '4', '5\+1': '6', '3\+2': '5', '5\+2': '7', '2\+1': '3',
                                     '6\+1': '7', 'Studio': '1', '7\+1': '8', '1\+1': '2', '4\+2': '6',
                                     '2\+2': '4', '8\+': '8', '6\+':'6', '8\+1':'9', '7\+':'7', '20 Above':'20',
                                     '9\+1': '10', '11\+1': '12', '12\+': '12', '10\+':'10', '1\+2': '3',
                                     '13\+': '13', '15\+' : '15', '9\+': '9'}, regex=True).astype(float)

df_floatRooms = df_shortLoc.copy()
df_floatRooms.Rooms = df_rooms
st.write(df_floatRooms.head())

st.subheader('Hapus nilai null dan duplikat')
df_clean = df_floatRooms.dropna()
df_clean = df_clean.drop_duplicates()

df_clean = df_clean.reset_index()
df_clean = df_clean.drop(['index'], axis=1)

st.write('Nilai null')
st.write(df_clean.isna().sum())
st.write('Nilai dupe')
st.write(df_clean.duplicated().sum())

st.subheader('Menghapus outlier')
df_outlier = df_clean.select_dtypes(exclude=['object'])
for column in df_outlier:
        f = plt.figure(figsize=(10,2))
        sns.boxplot(data=df_outlier, x=column)
        st.pyplot(f)

q1 = df_clean.select_dtypes(exclude=['object']).quantile (0.25)
q3 = df_clean.select_dtypes(exclude=['object']).quantile (0.75)
iqr = q3-q1

batas_bawah = q1 - (1.5 * iqr)
batas_atas = q3 + (1.5 * iqr)

df_noout = df_clean[~((df_clean.select_dtypes(exclude="object") < q1 - 1.5 * iqr) | (df_clean.select_dtypes(exclude="object") > q3 + 1.5 * iqr)).any(axis=1)]

df_ex_object = df_noout.select_dtypes(exclude=['object'])
outlier_filter = ((df_ex_object < batas_bawah) | (df_ex_object > batas_atas))

st.write('Jumlah non outlier setiap kolom')
for col in outlier_filter.columns :
    if df_noout[col].dtype != object :
        st.write('Nama Kolom:',col)
        st.write(outlier_filter[col].value_counts())

st.subheader('Visualisasi Data')
df_display = df_noout.copy()

Location_Count = df_display.Location.value_counts()

f = plt.figure(figsize=(10,5))

myColors = sns.color_palette('viridis')
Location_Count.plot(kind='bar',color=myColors)
plt.ylabel('Jumlah Properti')
plt.title('Jumlah Properti Per Lokasi')
plt.show()

st.pyplot(f)

plot = df_display['Furnishing'].value_counts().plot.pie(y=df_display['Furnishing'], figsize=(5, 5))
st.pyplot(plot.figure)

st.subheader('Encoding Categorical')

df_label = df_noout.copy()
df_label = df_label.drop(['Location', 'Property Type'], axis=1)

col_encode = ['Furnishing']

result = preprocessing.LabelEncoder()

df_label[col_encode] = df_label[col_encode].apply(result.fit_transform)

st.write(df_label.head())

st.header('--==Data Modelling==--')
df_final = df_label.copy()
st.write(df_final.describe())

X = df_final.iloc[:,0:-1].values
y = df_final.iloc[:,-1].values

train_X,validation_X, train_y,validation_y = train_test_split(X,y,random_state=0)
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X,train_y)

diamonds_pred = forest_model.predict(validation_X)
# rounds up the mean absolute error to two decimal places
mse = round(mean_absolute_error(validation_y,diamonds_pred),2)

st.write('Mean Absolute Error:', mse)

rmse = np.sqrt(mse)

st.write('Root Mean Absolute Error:', rmse)