import pandas as pd
from pathlib import Path

import streamlit as st
import sklearn
from sklearn.decomposition import PCA


# データの読み込み
path_parent = Path(__file__).resolve().parent.parent
path_data = path_parent.joinpath('data/whisky.csv')
df = pd.read_csv(path_data)

# 分析用にデータを整形・表示
df_pca = df.set_index('Distillery',inplace=False)
df_pca = df_pca.iloc[:, 1:13]
ncol_df_pca = len(df_pca.columns)
st.header('データ')
st.dataframe(df_pca)

# テイストごとの分布を表示
st.header('テイストごとの分布')
col1, col2, col3 = st.columns(3)
for i in range(ncol_df_pca):
    taste = df_pca.columns[i]
    j = i % 3
    if j == 0:
        with col1:
            st.write(taste)
            st.bar_chart(df[taste].value_counts(ascending=True))
    elif j == 1:
        with col2:
            st.write(taste)
            st.bar_chart(df[taste].value_counts(ascending=True))
    else:
        with col3:
            st.write(taste)
            st.bar_chart(df[taste].value_counts(ascending=True))

# 主成分分析の実行
st.header('主成分分析')
pca = PCA()
pca.fit(df_pca)

# 寄与率の表示
df_cr = pd.DataFrame(pca.explained_variance_ratio_, index=[i + 1 for i in range(ncol_df_pca)])
st.subheader('寄与率')
st.bar_chart(df_cr)

# 固有ベクトルの表示
df_eigenvector = pd.DataFrame(pca.components_, columns=df_pca.columns, index=["PC{}".format(x + 1) for x in range(ncol_df_pca)])
st.subheader('固有ベクトル')
st.dataframe(df_eigenvector)

# 固有ベクトルの散布図を表示
st.subheader('固有ベクトルの散布図')
st.scatter_chart(
    df_eigenvector.T.reset_index(),
    x='PC1',
    y='PC2',
    color='index'
)

# 主成分得点の表示
feature = pca.transform(df_pca)
df_pcs = pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(ncol_df_pca)])
df_pcs.insert(0, 'Distillery', df['Distillery'])
st.subheader('主成分得点')
st.dataframe(df_pcs.set_index('Distillery',inplace=False))

# 主成分得点の散布図を表示
st.subheader('主成分得点の散布図')
st.scatter_chart(
    df_pcs,
    x='PC1',
    y='PC2',
    color='Distillery'
)