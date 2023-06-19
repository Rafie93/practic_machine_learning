import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


dataset = pd.read_excel('dataset-penghasilan.xlsx')
print(dataset)

plt.scatter(dataset['jumlah_anggota'], dataset['penghasilan_(juta)'])
plt.xlabel('jumlah_anggota')
plt.ylabel('penghasilan_(juta)')
plt.show()

scaler = StandardScaler()
scaler.fit(dataset)
dataset_scaled = scaler.transform(dataset)
dataset_scaled = pd.DataFrame(dataset_scaled,
                              columns=['jumlah_anggota','penghasilan_(juta)'])
print(dataset_scaled)

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(dataset_scaled[['jumlah_anggota',
                                             'penghasilan_(juta)'
                                             ]])
dataset['tipe_penghasilan'] = y_predicted
print(dataset)

df1 = dataset[dataset.tipe_penghasilan == 0]
df2 = dataset[dataset.tipe_penghasilan == 1]
df3 = dataset[dataset.tipe_penghasilan == 2]

plt.scatter(df1.jumlah_anggota, df1['penghasilan_(juta)'], color='green')
plt.scatter(df2.jumlah_anggota, df2['penghasilan_(juta)'], color='red')
plt.scatter(df3.jumlah_anggota, df3['penghasilan_(juta)'], color='black')

plt.xlabel('jumlah_anggota')
plt.ylabel('penghasilan_(juta')
plt.grid()
plt.show()

conditions = [
    (dataset['tipe_penghasilan']==0),
    (dataset['tipe_penghasilan']==1),
    (dataset['tipe_penghasilan']==2)]
choices = ['Tinggi','Rata-rata','Rendah']
dataset['tipe_penghasilan'] = np.select(conditions, choices)
print(dataset)