mport pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv(r'C:\Users\221801049\Iris.csv')
df.head(150)
df.shape
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv(r'C:\Users\221801049\Iris.csv')
df.head(150)
df.shape
	df_Setosa = df[df['Species'] == 'Iris-setosa']
df_Virginica = df[df['Species'] == 'Iris-virginica']
df_Versicolor = df[df['Species'] == 'Iris-versicolor']
plt.figure(figsize=(10, 6))
plt.scatter(df_Setosa['SepalWidthCm'], np.zeros_like(df_Setosa['SepalWidthCm']), label='Setosa',                                                                                                                                                alpha=0.6)
plt.scatter(df_Versicolor['SepalWidthCm'], np.zeros_like(df_Versicolor['SepalWidthCm']),                                                                                                                 label='Versicolor', alpha=0.6)
plt.scatter(df_Virginica['SepalWidthCm'], np.zeros_like(df_Virginica['SepalWidthCm']),                                                                                                                   label='Virginica', alpha=0.6)
plt.xlabel('Sepal Width')
plt.title('Sepal Width Distribution by Iris Variety')
plt.legend()
plt.show()      
plt.figure(figsize=(10, 6))
plt.scatter(df_Setosa['SepalLengthCm'], np.zeros_like(df_Setosa['SepalLengthCm']),  
                                                                                                                         label='Setosa',alpha=0.6)
plt.scatter(df_Versicolor['SepalLengthCm'], np.zeros_like(df_Versicolor['SepalLengthCm']), 
                                                                                                                         label='Versicolor', alpha=0.6)
plt.scatter(df_Virginica['SepalLengthCm'], np.zeros_like(df_Virginica['SepalLengthCm']), 
                                                                                                                         label='Virginica',  alpha=0.6)
plt.xlabel('Sepal Length')
plt.title('Sepal Length Distribution by Iris Variety')
plt.legend()
plt.show()
plt.figure(figsize=(12, 6))
plt.scatter(df_Setosa['PetalWidthCm'], np.zeros_like(df_Setosa['PetalWidthCm']), label='Setosa', alpha=0.6)
plt.scatter(df_Versicolor['PetalWidthCm'], np.zeros_like(df_Versicolor['PetalWidthCm']), label='Versicolor', alpha=0.6)
plt.scatter(df_Virginica['PetalWidthCm'], np.zeros_like(df_Virginica['PetalWidthCm']), label='Virginica', alpha=0.6)
plt.xlabel('Petal Width (cm)')
plt.title('Petal Width Distribution by Iris Variety')
plt.legend()
plt.show()
plt.figure(figsize=(12, 6))
plt.scatter(df_Setosa['PetalLengthCm'], np.zeros_like(df_Setosa['PetalLengthCm']), label='Setosa', alpha=0.6)
plt.scatter(df_Versicolor['PetalLengthCm'], np.zeros_like(df_Versicolor['PetalLengthCm']), label='Versicolor',                                                                                                                                                alpha=0.6)
plt.scatter(df_Virginica['PetalLengthCm'], np.zeros_like(df_Virginica['PetalLengthCm']), label='Virginica',                                                                                                                                                alpha=0.6)
plt.xlabel('Petal Length (cm)')
plt.title('Petal Length Distribution by Iris Variety')
plt.legend()
plt.show()
g = sns.FacetGrid(df, hue='Species', height=5)
g.map(plt.scatter, 'SepalWidthCm', 'PetalWidthCm', alpha=0.6)
g.add_legend()
plt.show()
g = sns.FacetGrid(df, hue='Species', height=5)
g.map(plt.scatter, 'SepalLengthCm', 'PetalLengthCm', alpha=0.6)
g.add_legend()
plt.show()
g=sns.pairplot(df,hue='Species',height=2)
plt.show()
