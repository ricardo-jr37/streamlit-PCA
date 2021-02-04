import streamlit as st
import pandas as pd
import numpy as np
import base64
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from bokeh.plotting import figure
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df = pd.read_csv('data/wine.data', sep = ',',names=range(1, 15, 1))

st.title('Principal Component Analysis (PCA)')
st.markdown("""
Esse é um aplicativo simples para analisarmos as componentes principais de um dataset de Classificação de Vinhos. Os dados são resultados de uma análise química de vinhos cultivados na mesma região da Itália, mas derivados de três cultivares diferentes. A análise determinou as quantidades de 13 constituintes encontrados em cada um dos três tipos de vinhos.
* **Python libraries:** pandas, numpy, scikit-learn and streamlit 
* **Data source:** [Dataset-Wine](https://archive.ics.uci.edu/ml/datasets/Wine)
""")

st.sidebar.header('Parâmetros do Modelo')

def user_input_features():
    num_componentes = st.sidebar.slider('Número de Componentes Principais: ',1,13,5)
    num_vizinhos = st.sidebar.slider('Número de Vizinhos mais próximos(KNN): ',1,10,5)
    return num_componentes, num_vizinhos

#num_comp = user_input_features()
    
st.header('Exibição do Dataset')
st.write("Dimensão dos dados {} linhas e {} colunas".format(str(df.shape[0]), str(df.shape[1])))
st.dataframe(df)

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download = "wine.csv">Dowload CSV File</a>'
    return href

st.markdown(filedownload(df), unsafe_allow_html=True)

#Mapa de calor
if st.button('Mapa de Calor'):
    st.header('Mapa de calor de correlação dos dados')
    st.write('Por esse mapa de calor a gente ver com as variáveis se relacionam entre si.')
    corr = df.drop(columns=[1]).corr()
    mask = np.zeros_like(corr)
    mask [np.triu_indices_from(mask)] = True
    with sns.axes_style('white'):
        f,ax=plt.subplots(figsize=(7,5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot(f)

#Dados Padronizados
scaled = df.to_numpy()
x = scaled[:, 1:14].copy()
y = scaled[:, 0].copy()
x = StandardScaler().fit_transform(x)
pca = PCA()
principalComponents = pca.fit_transform(x)
names = []
for i in range(1, 14, 1):
    names.append('PC {}'.format(i))
names.append('Target')
df_autovetores=pd.DataFrame(principalComponents)
finalDf = pd.concat([df_autovetores, df [[1]]], axis = 1)
finalDf.columns = names

#Autovalores e autovetore
if st.button('Principais Componentes'):
    st.header('Dataset em função das suas componentes principais')
    st.write('Quando padronizamos os nossos dados, nós fazemos duas operações. Centralizamos o espaço de dados, para evidenciar as relações lineares que existem entre os atributos. E, fazemos o reescalonamento dos dados, para as medidas de escalas não atrapalharem na hora da produção do nosso modelo.')
    st.dataframe(finalDf)

#Variância explicada
if st.button('Variância explicada'):
    st.header('Soma acumulada das variâncias')
    st.write('Toda componente principal tem um autovalor associado, as componentes de mais importância tem os maiores autovalores. A soma dos autovalores é igual soma acumulada da variância dos nossos dados originais. Desse modo, podemos diminuir a dimensão do nosso dataset sem perder muita informação ao definir de quanto de informação queremos. Quando diminuimos a dimensão do nosso dataset, pode facilitar na análise dos nossos dados e diminuir redudâncias de dados.')
    matrizcov = np.corrcoef(x, rowvar=0)
    autovalores, _ = np.linalg.eig(matrizcov)
    idx = autovalores.argsort()[::-1]   
    autovalores = autovalores[idx]
    soma_acumulada = []
    soma_auto = np.sum(autovalores)
    soma = 0
    for i in range(0, len(autovalores)):
        if len(soma_acumulada) == 0:
            soma+=(autovalores[i]/soma_auto)*100
            soma_acumulada.append(soma)
        else:
            soma+=(autovalores[i]/soma_auto)*100
            soma_acumulada.append(soma)
    df_var_explicada= pd.DataFrame(soma_acumulada, columns=['VariânciaExplicada'])
    df_var_explicada.index = range(1,14,1)
    p = figure(
        title='Soma acumulada das importâncias de cada componente principal:',
        x_axis_label='Número de Componentes',
        y_axis_label='Variância Explicada(%)')
    
    
    p.scatter(df_var_explicada.index, df_var_explicada['VariânciaExplicada'], line_width=2)
    #st.pyplot(fig)
    st.bokeh_chart(p, use_container_width=True)

X = finalDf.drop(columns=['Target']).copy()
Y = finalDf['Target'].copy()
#Número de componentes principais que eu vou usar
num_pcs, num_vizinho = user_input_features()

x_train, x_test, y_train, y_test = train_test_split(X[:][names[:num_pcs]], Y, test_size = 0.30)
knn = KNeighborsClassifier(n_neighbors=num_vizinho)
#Train the model using the training sets
knn.fit(x_train, y_train)
#Predict the response for test dataset
y_pred = knn.predict(x_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct
st.header('Criando um modelo usando PCA')
st.write("""
Para fins didáticos você pode passar alguns parâmetros para criarmos um modelo de machine learning. Você pode passar o número de componentes principais, a fim de visualizar como o número de dimensões pode afetar a criação do nosso modelo. Além disso, vamos usar o algoritmo K-Nearest Neighbors(KNN) para criarmos o nosso modelo, assim, pode definir o número de vizinhos mais próximos que o algoritmo tem que usar para criar um modelo para as componentes principais e o dataset original. Assim, vamos comparar as acurácias entre as componentes principais e o dataset original. 
""")
st.subheader('Performance do dataset usando {} Componentes Principais'.format(num_pcs))
st.write("Acurácia: ",metrics.accuracy_score(y_test, y_pred))

#dataset original
X2 = df.drop(columns=[1]).copy()
Y2 = df[1].copy()
x_train2, x_test2, y_train2, y_test2 = train_test_split(X2, Y2, test_size = 0.30)
knn2 = KNeighborsClassifier(n_neighbors=num_vizinho)
#Train the model using the training sets
knn2.fit(x_train2, y_train2)
#Predict the response for test dataset
y_pred2 = knn2.predict(x_test2)
st.subheader('Performance do dataset original usando {} vizinhos'.format(num_vizinho))
st.write("Acurácia: ",metrics.accuracy_score(y_test2, y_pred2))
