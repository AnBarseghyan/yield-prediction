#!/usr/bin/env python
# coding: utf-8

# In[1735]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
pd.set_option("display.max_columns", None)



def create_data(filename):
    pkl_file = open(filename, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    
    # remove features with 0 value, 
    data = data.drop(columns = ['ndvi_hist_0', 'planter_hist_0', 'planter_hist_1', 
                            'planter_hist_6', 'planter_hist_7', 'planter_hist_8'])
    #remove not  necessary columns
    data = data.drop(columns=['crop_type', 'field_info', 'px_num', 'tile_number', 'valid_px_proportion'])
    return data
    
    

def corr_matrix(data, feature_list):
    plt.figure(figsize=(12,6))
    cor = data[feature_list].corr()
    sns.heatmap(cor, annot=True)
    plt.title('correlation between features and target variable')
    plt.show()


# In[1564]:


def hist_feature(data, target_name, title = 'all data'):
    sns.displot(data, x=target_name, binwidth=10)
    plt.title(f'{target_name} distribution, {title}')


# In[563]:


def plot_feature_scatter(data, feature_name, target_name):
    sns.regplot(x=feature_name,
            y=target_name, 
            data=data,  scatter_kws={"color": "yellow"}, line_kws={"color": "blue"})
    return f'linear connection between {feature_name} and {target_name}'


# In[575]:


#analysis of harvest_mean for different value of the feature
def plot_features_dist(df, feature_name, target_name):
    plot_table = {}
    perc = -1
    for i in range(0, 100, 10):
        limit_down = np.percentile(df[feature_name], i)
        limit_up = np.percentile(df[feature_name], i+10)
        if limit_down > perc:
            plot_table[np.round(limit_down, 2)] = int(df[(df[feature_name] >= limit_down) & 
                                    (df[feature_name] <= limit_up)][target_name].mean())
            perc = limit_down
    courses = list(plot_table.keys())
    values = list(plot_table.values())
  
    fig, ax = plt.subplots()
    
    bar_height = list(plot_table.values())
    bar_x = list(range(1, len(bar_height)+1))
    bar_tick_label = list(plot_table.keys())
    bar_label = list(plot_table.values())

    bar_plot = plt.bar(bar_x,bar_height,tick_label=bar_tick_label)

    def autolabel(rects):
        for idx,rect in enumerate(bar_plot):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    bar_label[idx],
                    ha='center', va='bottom', rotation=0)

    autolabel(bar_plot)

    plt.ylim(50,300)

    plt.title(f'harvest_mean for different value of {feature_name}')
    plt.ylabel(f'{target_name}, mean value')
    plt.xlabel(feature_name)
    plt.show()


# In[591]:


def plot_distribution(data, feature_name, target_name):
    data['feature'] = np.where(data[feature_name] == 0, '[0]', 
                         np.where((data[feature_name]  > 0) & (data[feature_name] <= 0.1), '(0 - 0.1]',
                                  np.where((data[feature_name] > 0.1) & (data[feature_name]  <= 0.4), '(0.1 - 0.4]', '> 0.4')))
    mean_values  = np.round(data.groupby('feature')[target_name].mean())
    keys = ['[0]','(0 - 0.1]', '(0.1 - 0.4]', '> 0.4']
    box = sns.boxenplot(x="feature", y=target_name, data=data, hue_order = keys, order = keys)
    box.set(xlabel=feature_name, title = f'{target_name} vs {feature_name}') 
    for i in keys:
        box.annotate(str(mean_values[i]), xy = (keys.index(i), mean_values[i]), color='black', horizontalalignment = 'center')
    plt.title(f"distribution of {target_name} for differnt value of {feature_name}")


# In[593]:


def plot_distribution_for_plant(data, feature_name, target_name):
    data['feature'] = np.where((data[feature_name]  > 25000) & (data[feature_name] <= 30000), '(25000 - 30000]',
                                  np.where((data[feature_name] > 30000) & (data[feature_name]  <= 33000), '(30000 - 33000]',
                                   np.where((data[feature_name] > 33000) & (data[feature_name]  <= 36000), '(33000 - 36000]','(36000 - 40000]')))
    mean_values  = np.round(data.groupby('feature')[target_name].mean())
    keys = ['(25000 - 30000]', '(30000 - 33000]', '(33000 - 36000]', '(36000 - 40000]']
    box = sns.boxenplot(x="feature", y=target_name, data=data, hue_order = keys, order = keys)
    box.set(xlabel=feature_name, title = f'{target_name} vs {feature_name}') 
    for i in keys:
        box.annotate(str(mean_values[i]), xy = (keys.index(i), mean_values[i]), color='black', horizontalalignment = 'center')
    plt.title(f"distribution of {target_name} for differnt value of {feature_name}")




def plot_distribution_for_groups(data, feature_name, target_name):
    table = {}
    data['feature'] = np.where(data[feature_name] == 0, '[0]', 
                         np.where((data[feature_name]  > 0) & (data[feature_name] <= 0.1), '(0 - 0.1]',
                                  np.where((data[feature_name] > 0.1) & (data[feature_name]  <= 0.4), '(0.1 - 0.4]', '> 0.4')))
    
    data['feature_target'] = np.where((data[target_name]  > 0) & (data[target_name] <= 150), '50-150',
                                  np.where((data[target_name] > 150) & (data[target_name]  <= 220), '150-220', 
                                           np.where((data[target_name] > 220) & (data[target_name]  <= 260), '220-260','>260')))
    keyss = ['[0]','(0 - 0.1]', '(0.1 - 0.4]', '> 0.4']
    for j in np.unique(data['feature_target']):
        listt = []
        for i in keyss:
            listt.append(data[data['feature'] == i].groupby('feature_target')[target_name].count()[j] /
                         data[data['feature'] == i]['feature'].count())
        table[j] = listt
    plotdata = pd.DataFrame({
    "50-150": table["50-150"],
    "150-220": table["150-220"],
    "220-260": table["220-260"],
    ">260": table[">260"]
    }, 
    index=['[0]','(0 - 0.1]', '(0.1 - 0.4]', '> 0.4'])
    plotdata.plot(kind="bar")
    plt.title(f'{target_name} groups for different value of {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel(f"{target_name}, count, %")


# In[1091]:


def plot_clusters(data, feature_list):
    with open("kmeans_model1.pkl", "rb") as f:
        kmeans_model = pickle.load(f)
    sc = pickle.load(open('scaler.pkl','rb'))
    scale_data = sc.transform(data[feature_list])
    data['cluster'] = kmeans_model.predict(scale_data)
    data_cluster={}
    for i in np.unique(data.cluster):
        data_cluster[i] = data[data['cluster'] == i]
    print(data_cluster.keys())
    table = {}
    for j in feature_list:
        listt = []
        for i in range(5):
            listt.append(data_cluster[i][j].mean())
        listt.append(data[j].mean())
        table[j] = listt
    table
    groups = ['cluster1', 'cluster2','cluster3', 'cluster4','cluster5', "All Data"]
    plotdata = pd.DataFrame({
    f'{feature_list[0]}': table[feature_list[0]],
    f'{feature_list[1]}': table[feature_list[1]],
    f'{feature_list[2]}': table[feature_list[2]],
    f'{feature_list[3]}': table[feature_list[3]]
    },
    index = groups)
    plotdata.plot(kind="bar")
    plt.title("Cluster Analysis")
    plt.ylabel('alert features, mean')
    axes2 = plt.twinx()
    axes2.plot(groups, table['harvest_mean'], color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
    axes2.set_ylabel('harvest_mean, mean')
    plt.ylim(180, 250)
# In[1445]:


def autolabel(rects, bar_label):
    for idx,rect in enumerate(rects):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height,
                bar_label[idx],
                ha='center', va='bottom', rotation=0)


# In[1446]:


def plot_sepeartaed_groups(data, feature_list):
    classes = []
    for i in feature_list:
        classes.append(data[data[i] == 1]['harvest_mean'].mean())
    classes.append(data['harvest_mean'].mean())
    X = np.arange(6)
    bar_plot = plt.bar(X, classes, color = 'green', width = 0.3)
    plt.xlabel('Groups')
    plt.ylabel('harvest_mean, mean')
    plt.ylim(150,250)
    plt.title('harvest_mean for new groups based on alerts feature')
    feature_list.append('all data')
    for i in range(6):
        plt.text(6, 200-(i*10), f'group{i} - {feature_list[i]}')
    autolabel(bar_plot, np.floor(classes))


# In[1449]:


def tabel_between_features(data, feature_name1, feature_name2):
    data['feature_1'] = np.where(data[feature_name1] == 0, '(0.0)', 
                         np.where((data[feature_name1]  > 0) & (data[feature_name1] <= 0.1), '(0.0 - 0.1]',
                                  np.where((data[feature_name1] > 0.1) & (data[feature_name1]  <= 0.4), '(0.1 - 0.4]', '> 0.4')))
    data['feature_2'] = np.where(data[feature_name2] == 0, '(0.0)', 
                         np.where((data[feature_name2]  > 0) & (data[feature_name2] <= 0.1), '[0.0 - 0.1]',
                                  np.where((data[feature_name2] > 0.1) & (data[feature_name2]  <= 0.4), '(0.1 - 0.4]', '> 0.4')))
    df = data.pivot_table('harvest_mean', index='feature_1', columns='feature_2', aggfunc='mean')
    sns.heatmap(df, annot=True, fmt=".1f")
    plt.ylabel(feature_name1)
    plt.xlabel(feature_name2)
    plt.title(f'harvest_mean distribution by {feature_name1} and {feature_name2}')
    plt.show()


# ### features importance: information gain based on entropy

# In[1120]:


from sklearn.feature_selection import mutual_info_regression
def feature_imp(X, y):
    importances = mutual_info_regression(X, y)
    feat_imp = pd.Series(importances, X.columns)
    feat_imp.plot(kind='barh', color='teal')
    plt.title('dependency between feature and target variable based on information gain')
    plt.show()


# ## outlier detection

# In[1482]:


def feature_outlier(data, feature_list):
    df=[]
    for i in feature_list:
        df.append(data[i])

    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(df, patch_artist = True, notch ='True', vert = 0)
    colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF00FF']

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    for flier in bp['fliers']:
        flier.set(marker ='D',
                  color ='#e7298a',
                  alpha = 0.5)

    ax.set_yticklabels(feature_list)

    plt.title("feature outlier")
    
    plt.show()


# In[1622]:


#OUTLIER REMOVAL
def iqr(df, cols, lq=.05, uq=.95):
    for col in cols:
        Q1 = df[col].quantile(lq)
        Q3 = df[col].quantile(uq)
        IQR = Q3 - Q1
        df = df[np.logical_and((Q1 - 1.5 * IQR) <= df[col],
                             df[col] <= (Q3 + 1.5 * IQR))]
    return df


# ### scaling and pca

# In[1656]:


def scaling(data):
    scale_norm = StandardScaler()
    return scale_norm.fit(data)


# In[1657]:


def pca_method(data, for_index, components=2):
    pca = PCA(n_components=components)
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(data = principalComponents, index=for_index)
    return pca, principalDf


# In[1676]:


def plot_pca(pca):
    plt.figure(figsize=(4, 4))
    var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
    lbls = [str(x) for x in range(1,len(var)+1)]
    plt.bar(x=range(1,len(var)+1), height = var, tick_label = lbls)
    print(np.round(var.sum()))
    plt.show()


# In[ ]:




