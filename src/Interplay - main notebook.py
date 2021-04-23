#!/usr/bin/env python
# coding: utf-8

# # The Interplay of Compile-time and Run-time Options for Performance Prediction
# 
# This notebook follows the order and produce all the figures depicted in the related submission, "The Interplay of Compile-time and Run-time Options for Performance Prediction"

# #### First, we import some libraries

# In[43]:


# for arrays
import numpy as np

# for dataframes
import pandas as pd

# plots
import matplotlib.pyplot as plt
# high-level plots
import seaborn as sns

# statistics
import scipy.stats as sc
# hierarchical clustering, clusters
from scipy.cluster.hierarchy import linkage, cut_tree, leaves_list
from scipy import stats
# statistical tests
from scipy.stats import mannwhitneyu

# machine learning library
# Principal Component Analysis - determine new axis for representing data
from sklearn.decomposition import PCA
# Random Forests -> vote between decision trees
# Gradient boosting -> instead of a vote, upgrade the same tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
# Decision Tree
from sklearn.tree import DecisionTreeRegressor, plot_tree
# To add interactions in linear regressions models
from sklearn.preprocessing import PolynomialFeatures
# Elasticnet is an hybrid method between ridge and Lasso
from sklearn.linear_model import LinearRegression, ElasticNet
# To separate the data into training and test
from sklearn.model_selection import train_test_split
# Simple clustering (iterative steps)
from sklearn.cluster import KMeans

# we use it to interact with the file system
import os
# compute time
from time import time

# statistics
import scipy.stats as sc
# hierarchical clustering, clusters
from scipy.cluster.hierarchy import linkage, cut_tree, leaves_list
from scipy import stats
# statistical tests
from scipy.stats import mannwhitneyu


# ### Import data

# In[2]:


data_dir = "../data/"
name_systems = ["x264", "xz", "nodejs", "poppler"]

data = dict()
default_data = dict()
inputs_name = dict()

for ns in name_systems:
    
    data_path = data_dir+ns+'/'
    
    list_dir = os.listdir(data_path)
    list_dir.remove('ctime_options.csv')
    list_dir.remove('default')

    inputs_name[ns] = os.listdir(data_path+list_dir[0])
    inputs = inputs_name[ns]
    
    for j in range(len(inputs)):
        for i in range(len(list_dir)):
            loc = data_path+list_dir[i]+'/'+inputs[j]
            data[ns, list_dir[i], j] = pd.read_csv(loc)
        
        default_data[ns, j] = pd.read_csv(data_path+'default/'+inputs[j])


# # RQ1

# # RQ1.1

# ### Compute some boxplot of runtime performances

# #### Figure 2a

# In[91]:


ns ="x264"
dim = "size"
inputs_index = 7

listDim = []
for i in range(len(list_dir)):
    df = data[ns, list_dir[i], inputs_index]
    listDim.append(df[dim]/1e6)

print()

transposed_listDim = []
for i in range(len(listDim[0])):
    transposed_listDim.append([listDim[j][i] for j in range(len(listDim))])


red_square = dict(markerfacecolor='r', marker='s')
plt.figure(figsize=(20,10))
plt.grid()
plt.scatter([k+1 for k in range(30)], [np.mean(l) for l in transposed_listDim[100:130]],
           marker="x", color = "black", alpha = 1, s = 20)
plt.boxplot(transposed_listDim[100:130], flierprops=red_square, 
          vert=True, patch_artist=True, #widths=0.25,
          boxprops=dict(facecolor=(0,0,1,0.5),linewidth=1,edgecolor='k'),
          whiskerprops = dict(linestyle='-.',linewidth=1.0, color='black'))

plt.xticks([k for k in range(1, 31) if k%10==0 or k==1],[k for k in range(101,131) if k%10==0 or k==101], 
           rotation='vertical', size = 25)

#plt.title("x264, Sports video, "+dim, size = 25)
plt.ylabel("Output video size (Megabytes)", size = 30)
plt.xlabel("Runtime configuration ids", size=30)
plt.yticks(size=25)
plt.savefig("../results/boxplot_"+ns+"_"+dim+".png")
plt.show()


# In[93]:


count_fail = 0
remaining_pvals = []
for i in range(len(listDim)):
    for j in range(len(listDim)):
        try:
            if i!=j:
                remaining_pvals.append(stats.wilcoxon(listDim[i], listDim[j])[1])
        except:
            count_fail+=1
print(count_fail/(len(listDim)**2-len(listDim))*100, "% failures")


# In[4]:


np.mean([np.percentile(distr,75)-np.percentile(distr,25) for distr in transposed_listDim])


# In[5]:


np.mean(transposed_listDim)


# In[74]:


ns ="xz"
dim = "size"
inputs_index = 8

listDim = []
for i in range(len(list_dir)):
    df = data[ns, list_dir[i], inputs_index]
    listDim.append(df[dim]/1e6)

transposed_listDim = []
for i in range(len(listDim[0])):
    transposed_listDim.append([listDim[j][i] for j in range(len(listDim))])
    
np.mean([np.percentile(distr,75)-np.percentile(distr,25) for distr in transposed_listDim])


# In[7]:


np.mean(transposed_listDim)


# In[77]:


ns ="poppler"
dim = "size"
inputs_index = 8

listDim = []
for i in range(len(list_dir)):
    df = data[ns, list_dir[i], inputs_index]
    listDim.append(df[dim]/1e6)

transposed_listDim = []
for i in range(len(listDim[0])):
    transposed_listDim.append([listDim[j][i] for j in range(len(listDim))])
    
np.mean([np.percentile(distr,75)-np.percentile(distr,25) for distr in transposed_listDim])


# In[78]:


np.mean(transposed_listDim)


# #### Figure 2b

# In[98]:


ns ="xz"
dim = "time"
inputs_index = 4

listDim = []
for i in range(len(list_dir)):
    df = data[ns, list_dir[i], inputs_index]
    listDim.append(df[dim])

print()

transposed_listDim = []
for i in range(len(listDim[0])):
    transposed_listDim.append([listDim[j][i] for j in range(len(listDim))])

plt.figure(figsize=(20,10))

#plt.title("xz, Reymont file, "+dim, size = 25)
plt.ylabel("Compression time (seconds)", size = 30)
plt.xlabel("Runtime configuration ids", size = 30)

plt.grid()
plt.scatter([k+1 for k in range(len(transposed_listDim))], [np.mean(l) for l in transposed_listDim],
           marker="x", color = "black", alpha = 1, s = 20)
plt.boxplot(transposed_listDim, flierprops=red_square, 
          vert=True, patch_artist=True, #widths=0.25,
          boxprops=dict(facecolor=(0,0,1,0.5),linewidth=1,edgecolor='k'),
          whiskerprops = dict(linestyle='-.',linewidth=1.0, color='black'))
plt.xticks([k for k in range(1,31) if k%3==0 or k==1],[k for k in range(1,31) if k%3==0 or k ==1], 
           rotation='vertical', size =25)
plt.yticks(size=25)
plt.savefig("../results/boxplot_"+ns+"_"+dim+".png")
plt.show()


# In[103]:


stats.wilcoxon(listDim[3], listDim[10])


# In[11]:


ns ="poppler"
dim = "time"
inputs_index = 8

listDim = []
for i in range(len(list_dir)):
    df = data[ns, list_dir[i], inputs_index]
    listDim.append(df[dim])

transposed_listDim = []
for i in range(len(listDim[0])):
    transposed_listDim.append([listDim[j][i] for j in range(len(listDim))])
    
np.mean([np.percentile(distr,75)-np.percentile(distr,25) for distr in transposed_listDim])


# In[12]:


np.mean(transposed_listDim)


# In[13]:


ns ="xz"
dim = "time"
inputs_index = 4

listDim = []
for i in range(len(list_dir)):
    df = data[ns, list_dir[i], inputs_index]
    listDim.append(df[dim])

transposed_listDim = []
for i in range(len(listDim[0])):
    transposed_listDim.append([listDim[j][i] for j in range(len(listDim))])
    
[np.percentile(distr,75)-np.percentile(distr,25) for distr in transposed_listDim][7]


# In[14]:


np.mean(transposed_listDim)


# #### Figure 2c

# In[15]:


ns ="x264"
dim = "time"
inputs_index = 3

listDim = []
for i in range(len(list_dir)):
    df = data[ns, list_dir[i], inputs_index]
    listDim.append(df[dim])

print()

transposed_listDim = []
for i in range(len(listDim[0])):
    transposed_listDim.append([listDim[j][i] for j in range(len(listDim))])

plt.figure(figsize=(20,10))

#plt.title("x264, Music video, "+dim, size = 30)
plt.ylabel("Encoding time (seconds)", size = 30)
plt.xlabel("Runtime configuration ids", size=30)

plt.grid()
plt.scatter([k+1 for k in range(30)], [np.mean(l) for l in transposed_listDim[100:130]],
           marker="x", color = "black", alpha = 1, s = 20)
plt.boxplot(transposed_listDim[100:130], flierprops=red_square, 
          vert=True, patch_artist=True, #widths=0.25,
          boxprops=dict(facecolor=(0,0,1,0.5),linewidth=1,edgecolor='k'),
          whiskerprops = dict(linestyle='-.',linewidth=1.0, color='black'))

plt.xticks([k for k in range(1, 31) if k%10==0 or k==1],[k for k in range(101,131) if k%10==0 or k==101], 
           rotation='vertical', size = 25)
plt.yticks(size=25)
plt.savefig("../results/boxplot_"+ns+"_"+dim+".png")
plt.show()


# In[16]:


[np.percentile(distr,75)-np.percentile(distr,25) for distr in transposed_listDim][100:130]


# In[17]:


[np.mean(distr) for distr in transposed_listDim][100:130]


# #### Figure 2d

# In[45]:


ns ="nodejs"
dim = "ops"
inputs_index = 2

listDim = []
for i in range(len(list_dir)):
    df = data[ns, list_dir[i], inputs_index]
    listDim.append(df[dim])

print()

transposed_listDim = []
for i in range(len(listDim[0])):
    transposed_listDim.append([listDim[j][i] for j in range(len(listDim))])

plt.figure(figsize=(20,10))

#plt.title("nodejs, fsfile script, operation rate", size = 25)
plt.ylabel("Operation rate (#op/second)", size = 30)
plt.xlabel("Runtime configuration ids", size=30)

plt.grid()
plt.scatter([k+1 for k in range(len(transposed_listDim))], [np.mean(l) for l in transposed_listDim],
           marker="x", color = "black", alpha = 1, s = 20)
plt.boxplot(transposed_listDim, flierprops=red_square, 
          vert=True, patch_artist=True, #widths=0.25,
          boxprops=dict(facecolor=(0,0,1,0.5),linewidth=1,edgecolor='k'),
          whiskerprops = dict(linestyle='-.',linewidth=1.0, color='black'))
plt.boxplot(transposed_listDim)
plt.xticks([k for k in range(1,31) if k%3==0 or k==1],[k for k in range(1,31) if k%3==0 or k ==1], 
           rotation='vertical', size =25)
plt.yticks(size=25)
plt.savefig("../results/boxplot_"+ns+"_"+dim+".png")
plt.show()


# In[46]:


np.mean([np.percentile(distr,75)-np.percentile(distr,25) for distr in transposed_listDim])


# In[47]:


np.mean(transposed_listDim)


# In[48]:


stats.wilcoxon(listDim[1], listDim[3])


# runtime dist of compile-time options 1 and 3 are not significantly different 

# In[50]:


stats.wilcoxon(listDim[1], listDim[11])


# runtime dist of compile-time options 1 and 11 are significantly different 

# # RQ1.2

# ### Compute the ratio between runtime performances of the compile-time options and the default configuration

# #### Table 2a - Average and standard deviation ratios

# In[21]:


def get_ratios(name_system, input_index, dim):
    
    list_inputs = inputs_name[name_system]
    
    nb_ctime_configs = len(os.listdir("../data/"+name_system))-2
    
    ratios = []
    
    for i in range(1, nb_ctime_configs+1):
        
        df = data[name_system, str(i), input_index]
        df_def = default_data[name_system, input_index]
        
        ratios.append(df[dim]/df_def[dim])
    
    return (np.round(np.mean(ratios),2), np.round(np.std(ratios),2))


# In[22]:


get_ratios("x264", 0, "size")


# In[23]:


results = dict()

perfs = dict()
perfs["x264"] = ["time", "fps"]
perfs["xz"] = ["time"]
perfs["poppler"] = ["time"]
perfs["nodejs"] = ["ops"]

for ns in name_systems:
    if ns in perfs:
        for p in perfs[ns]:                
            nb_inputs = len(inputs_name[ns])
            for input_index in range(nb_inputs):
                results[ns, input_index+1, p] = get_ratios(ns, input_index, p)

print("\\begin{tabular}{|c|c|c|c|c|c|}")
perfs = sorted(pd.Series([(k[0], k[2]) for k in results.keys()]).unique())
print("\\hline")
print("System")
print("& nodejs")
print("& poppler")
print("& \\multicolumn{2}{|c|}{x264}")
print("& xz")
#for i in range(len(perfs)):
#    print("& "+perfs[i][0])
print("\\\\ \\hline")
print("Perf. $\mathcal{P}$")
for i in range(len(perfs)):
    print("& "+perfs[i][1])
print("\\\\ \\hline")
for i in range(1, 13):
    print("$\mathcal{I}$\\#"+str(i))
    for j in range(len(perfs)):
        if i <= len(inputs_name[perfs[j][0]]):
            if results[perfs[j][0], i, perfs[j][1]]:
                mean_ratios, std_ratios = results[perfs[j][0], i, perfs[j][1]]
                print("& "+str(mean_ratios)+" $\pm$ "+str(std_ratios))
        else:
            print("& \\cellcolor[HTML]{C0C0C0}")
    print("\\\\ \\hline")
print("\\end{tabular}")


# #### Table 2b - Best ratios (minimal time, and max fps or operation per second)

# In[24]:


def get_ratios_indic(name_system, input_index, dim, indic):
    
    list_inputs = inputs_name[name_system]
    
    nb_ctime_configs = len(os.listdir("../data/"+name_system))-2
    
    ratios = []
    
    for i in range(1, nb_ctime_configs+1):
        
        df = data[name_system, str(i), input_index]
        df_def = default_data[name_system, input_index]
        
        ratios.append(df[dim]/df_def[dim])
    
    if indic == "max":
        res = np.round(np.max(ratios), 2)
    if indic == "min":
        res = np.round(np.min(ratios), 2)
    
    return res

results = dict()

perfs = dict()
perfs["x264"] = ["time", "fps"]
perfs["xz"] = ["time"]
perfs["poppler"] = ["time"]
perfs["nodejs"] = ["ops"]

for ns in name_systems:
    if ns in perfs:
        for p in perfs[ns]:
            nb_inputs = len(inputs_name[ns])
            if p == "time":
                for input_index in range(nb_inputs):
                    results[ns, input_index+1, p] = get_ratios_indic(ns, input_index, p, "min")
            else:
                for input_index in range(nb_inputs):
                    results[ns, input_index+1, p] = get_ratios_indic(ns, input_index, p, "max")
                    
print("\\begin{tabular}{|c|c|c|c|c|c|}")
perfs = sorted(pd.Series([(k[0], k[2]) for k in results.keys()]).unique())
print("\\hline")
print("System")
print("& nodejs")
print("& poppler")
print("& \\multicolumn{2}{|c|}{x264}")
print("& xz")
#for i in range(len(perfs)):
#    print("& "+perfs[i][0])
print("\\\\ \\hline")
print("Perf. $\mathcal{P}$")
for i in range(len(perfs)):
    print("& "+perfs[i][1])
print("\\\\ \\hline")
for i in range(1, 13):
    print("$\mathcal{I}$\\#"+str(i))
    for j in range(len(perfs)):
        if i <= len(inputs_name[perfs[j][0]]):
            if results[perfs[j][0], i, perfs[j][1]]:
                print("& "+str(results[perfs[j][0], i, perfs[j][1]]))
        else:
            print("& \\cellcolor[HTML]{C0C0C0}")
    print("\\\\ \\hline")
print("\\end{tabular}")


# # RQ2

# # RQ2.1

# ### Spearman correlogram

# In[25]:


# We define a function to plot the correlogram
def plot_correlationmatrix_dendogram(ns, dim):
    # ns : name_system
    # dim : dimension
    # output : a plot of an ordered correlogram of the different compile-time options
    
    # number of videos
    nb_ctime = len(os.listdir(data_dir+ns))-2
    
    for input_index in range(len(inputs_name[ns])):
        
        # matrix of coorelations
        corr = [[0 for x in range(nb_ctime)] for y in range(nb_ctime)]

        for i in range(nb_ctime):
            for j in range(nb_ctime):
                # A distribution of bitrates will have a correlaiton of 1 with itself
                if (i == j):
                    corr[i][j] = 1
                else:
                    # we compute the Spearman correlation between the input video i and the input video j
                    corr[i][j] = sc.spearmanr(data[ns, str(i+1), input_index][dim],
                                              data[ns, str(j+1), input_index][dim]).correlation
                    #corr[i][j] = np.corrcoef(data[ns, str(i+1), input_index][dim],
                    #                                data[ns, str(j+1), input_index][dim])[0,1]
                    

        # we transform our matrix into a dataframe
        df = pd.DataFrame(corr)

        # group the videos, we choose the ward method 
        # single link method (minimum of distance) leads to numerous tiny clusters
        # centroid or average tend to split homogeneous clusters
        # and complete link aggregates unbalanced groups. 
        links = linkage(df, method="ward",)
        order = leaves_list(links)

        # we order the correlation following the aggregation clustering
        mask = np.zeros_like(corr, dtype=np.bool)

        for i in range(nb_ctime):
            for j in range(nb_ctime):
                # Generate a mask for the upper triangle
                if i>j:
                    mask[order[i]][order[j]] = True
        
        g = sns.clustermap(df, cmap="vlag", mask=mask, method="ward",
                       linewidths=0, figsize=(13, 13), #cbar_kws={"ticks":ticks}, 
                       vmin =-1)
        g.ax_heatmap.set_yticklabels([])
        #g.ax_heatmap.set_xticklabels([])
        g.ax_heatmap.tick_params(right=False, bottom=False)
        # abcissa and ordered labels
        g.ax_heatmap.set_xlabel("Compile-time options", fontsize = 18)
        g.ax_heatmap.set_ylabel("Compile-time options", fontsize = 18)
        # we save the figure in the result folder
        plt.savefig("../results/"+ns+"/"+"corr_"+str(input_index+1)+"_"+dim+".png")
        # we show the graph
        plt.show()


# In[26]:


plot_correlationmatrix_dendogram("nodejs", "ops")


# In[27]:


#plot_correlationmatrix_dendogram("x264", "size")
#plot_correlationmatrix_dendogram("xz", "size")
#plot_correlationmatrix_dendogram("poppler", "size")


# In[28]:


ns = "x264"
dim = "time"
input_index = 7

for input_index in range(8):
    # number of videos
    nb_ctime = len(os.listdir(data_dir+ns))-2

    # matrix of correlations
    corr = [[0 for x in range(nb_ctime)] for y in range(nb_ctime)]

    for i in range(nb_ctime):
        for j in range(nb_ctime):
            if (i == j):
                corr[i][j] = 1
            else:
                corr[i][j] = sc.spearmanr(data[ns, str(i+1), input_index][dim],
                                          data[ns, str(j+1), input_index][dim]).correlation

    print(np.min(corr))


# In[29]:


ns = "xz"
dim = "time"

for input_index in range(8):
    # number of videos
    nb_ctime = len(os.listdir(data_dir+ns))-2

    # matrix of correlations
    corr = [[0 for x in range(nb_ctime)] for y in range(nb_ctime)]

    for i in range(nb_ctime):
        for j in range(nb_ctime):
            if (i == j):
                corr[i][j] = 1
            else:
                corr[i][j] = sc.spearmanr(data[ns, str(i+1), input_index][dim],
                                          data[ns, str(j+1), input_index][dim]).correlation

    print(np.min(corr))


# In[30]:


ns = "x264"
dim = "time"

for input_index in range(8):
    # number of videos
    nb_ctime = len(os.listdir(data_dir+ns))-2

    # matrix of correlations
    corr = [[0 for x in range(nb_ctime)] for y in range(nb_ctime)]

    for i in range(nb_ctime):
        for j in range(nb_ctime):
            if (i == j):
                corr[i][j] = 1
            else:
                corr[i][j] = sc.spearmanr(data[ns, str(i+1), input_index][dim],
                                          data[ns, str(j+1), input_index][dim]).correlation

    print(np.min(corr))


# In[31]:


ns = "nodejs"
dim = "ops"
input_index = 9

# number of videos
nb_ctime = len(os.listdir(data_dir+ns))-2

# matrix of correlations
corr = [[0 for x in range(nb_ctime)] for y in range(nb_ctime)]

for i in range(nb_ctime):
    for j in range(nb_ctime):
        if (i == j):
            corr[i][j] = 1
        else:
            corr[i][j] = sc.spearmanr(data[ns, str(i+1), input_index][dim],
                                      data[ns, str(j+1), input_index][dim]).correlation


# In[32]:


np.mean(corr[16][24])


# In[33]:


np.mean(corr[40][23])


# In[34]:


np.mean(corr[27][13])


# In[35]:


np.mean(corr[29][8])


# ### Binary tree

# In[36]:


perfs = dict()

perfs["nodejs"] = ["ops"]
perfs["poppler"] = ["time", "size"]
perfs["x264"] = ["kbs", "fps", "size", "time", "frames"]
perfs["xz"] = ["time", "size"]

def aggregate_data(ns, input_index, dim):
    
    nb_ctime = len(os.listdir(data_dir+ns))-2
    
    ctime_data = pd.read_csv(data_dir+ns+"/ctime_options.csv")
    
    # we delete the other perfs to avoid ocnsidering them as predicting variables
    to_delete_perfs = list(perfs[ns])
    to_delete_perfs.remove(dim)
    to_delete_perfs.append('configurationID')
    
    aggreg_vals = []

    for index_comp in range(nb_ctime):
        
        val = ctime_data.iloc[index_comp][1:]
        
        df_runtime = data[ns, str(index_comp+1), input_index]
        df_runtime = df_runtime.drop(to_delete_perfs, axis = 1)
        
        df_runtime_modif = pd.get_dummies(df_runtime.drop([dim], axis=1))
        df_runtime_modif[dim] = df_runtime[dim]
        
        for rt_config_id in range(df_runtime.shape[0]):
            aggreg_vals.append(list(tuple(val) + tuple(df_runtime_modif.loc[rt_config_id])))
        
    res_df = pd.DataFrame(aggreg_vals)
    res_df.columns = list(tuple(ctime_data.columns[1:]) + tuple(df_runtime_modif.columns))
    
    return res_df

def draw_tree(ns, input_index, dim, max_depth):
    
    res_df = aggregate_data(ns, input_index, dim)

    y = res_df[dim]
    X = res_df.drop([dim], axis=1)

    dt = DecisionTreeRegressor(max_depth = max_depth)
    dt.fit(X,y)

    plt.figure(figsize=(20,20))
    plot_tree(dt, feature_names=res_df.columns, filled=True)
    plt.savefig("../results/"+ns+"/tree_input_"+str(input_index+1)+"_"+dim+".png")
    plt.show()


# In[37]:


for i in range(len(inputs_name["nodejs"])):
    draw_tree("nodejs", i, "ops", 3)


# ### Feature importances

# In[38]:


def show_imp(ns, input_index, dim, col_names, color):
    
    res_df = aggregate_data(ns, input_index, dim)

    y = res_df[dim]
    X = res_df.drop([dim], axis=1)

    rf = RandomForestRegressor()
    rf.fit(X,y)
    
    res_imp = pd.Series(rf.feature_importances_, res_df.columns[:-1])
    res_imp
    
    plt.figure(figsize = (20,10))
    plt.grid()
    plt.ylabel("Random Forest importance (%)", size = 20)
    plt.yticks(size=15)
    plt.bar(range(len(res_imp.values)), 100*res_imp, color= color)
    if col_names:
        plt.xticks(range(len(res_imp.values)), col_names, rotation=45, size =15)
    else:
        plt.xticks(range(len(res_imp.values)), res_imp.index, rotation=45, size =15)
    plt.savefig("../results/"+ns+"/rf_input_"+str(input_index+1)+"_"+dim+".png")
    plt.show()


# In[39]:


col_node = ['--cross-comp', '--fully-static', '--enable-lto',
       '--ossl-no-asm', '--ossl-is-fips',
       '--eepc', '--wout-intl',
       '--without-ns', '--wout-cache',
       '--en-static', '--v8-lite-m', 'jitless',
       'xp-wasm', 'xp-vm',
       'p-symlinks', 'no-warnings', 'mem-debug']

color = ["salmon"]*11+["darkgreen"]*6

for i in range(len(inputs_name["nodejs"])):
    show_imp("nodejs", i, "ops", col_node, color)


# In[ ]:





# # RQ2.2

# In[40]:


ns = "nodejs"
dim = "ops"

res_rq22 = dict()

#learning rates
lrs = [0.01, 0.05, 0.1]

for input_index in range(len(inputs_name[ns])):

    df = aggregate_data(ns, input_index, dim)

    perf_val = df.iloc[np.argmax(df[dim])][dim]

    val_default = default_data[ns, input_index][dim][0]

    res_rq22[input_index, "Oracle"] = perf_val/val_default
    
    print("Input="+str(input_index+1)+" : "+str(perf_val/val_default))


# In[41]:


for lr in lrs:
    
    for input_index in range(len(inputs_name[ns])):

        res_val = []
        res_val_def = []

        for i in range(10):

            df = aggregate_data(ns, input_index, dim)

            y = df[dim]
            X = df.drop([dim], axis=1)

            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = lr)

            rf = RandomForestRegressor()
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)

            if np.max(y_train) > np.max(y_pred):
                runtime_values = X_train.iloc[np.argmax(y_train)]
            else:
                runtime_values = X_test.iloc[np.argmax(y_pred)]

            perf_val = df.iloc[runtime_values.name][dim]

            res_default = default_data[ns, input_index]
            # [0] = default value, see the configuration csvs if you are not sure
            val_default = res_default[dim][0]
            val_mean = np.mean(res_default[dim])

            #res_default.columns = ['val'+str(k) for k in range(len(res_default.columns))]
            #for i in range(1,len(res_default.columns)-1):
            #    query_feature = "val"+str(i)+"=="+str(runtime_values[10+i])
            #    res_default = res_default.query(query_feature)


            res_val.append(perf_val/val_mean)
            res_val_def.append(perf_val/val_default)

        #print("Input " + str(input_index) +" : "+ str(np.mean(res_val)))
        res_rq22[input_index, str(lr)] = np.mean(res_val_def)


# In[42]:


lrs2 = [str(lr) for lr in lrs]
lrs2.append("Oracle")
print("\\begin{table}[htb]")
print("\\begin{tabular}{|c|c|c|c|c|}")
print("\\hline")
print("Training Size")
for lr in lrs2:
    print("& "+lr)
print("\\\\ \\hline")
for input_index in range(10):
    print("$\mathcal{I}$\\#"+str(input_index+1))
    for lr in lrs2:
        print("& "+str(np.round(res_rq22[input_index,lr],3)))
    print("\\\\ \\hline")
print("\\end{tabular}")
print("\\label{tab:cross-tuning}")
print("\\caption{Performance ratios between the best predicted configuration and the default configuration}")
print("\\end{table}")


# In[ ]:




