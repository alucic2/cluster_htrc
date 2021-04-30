#!/usr/bin/env python
# coding: utf-8

# ### Identifying boundaries of the main content of fiction and non-fiction works and separating it from paratextual elements using an unsupervised clustering method

# In[1]:


import re
import os
import glob
import pandas as pd
import collections
from collections import Counter
import numpy as np
import htrc_features
from htrc_features import Volume
from htrc_features import utils
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn


# In[2]:


volume_pages = Volume('msu.31293015784899') #HathiTrust id
#ucw.ark:/13960/t07w6r028 - Tom Sawyer edition
#hvd.hn6mhj - David Copperfield edition
#mdp.39015042089402 - House of Mirth edition
#msu.31293015784899 - Jane Eyre


# In[16]:


d=[]
for page in volume_pages:
    tokens = page.tokenlist(drop_section=True,pos=False, drop_page=False)
    tokens1 = tokens.reset_index() #flattening the file
    tokens2 = tokens1[['token', 'count']] #only keeping token and count from extracted features dataset
    token = tokens1['token'].to_list() #converting to list
    count = tokens1['count'].to_list()
    d2 = dict(zip(token, count)) #creating a dictionary format
    d.append(d2)


# In[ ]:


pages =[]
pgs = enumerate(d, start=1) #getting the page numbers by enumerating the token:count pairs in d. Some page numbers are missing from extracted features dataset 
for i in pgs:
    pages.append(i[0]) 


# ### initializing DictVectorizer()

# In[18]:


v = DictVectorizer(sparse=False)


# ### Token counts on each page of the book are converted into a document matrix

# In[19]:


dtm = v.fit_transform(d)  


# ### Pairwise cosine similarity is calculated and distance between each pair of pages is obtained by subtracting cosine similarity from 1

# In[44]:


dist = 1 - cosine_similarity(dtm)# running pairwise cosine similarity on dtm object and subtracting from 1


# ### Multi dimensional scaling allows viewing of distances between pages in two dimensions

# In[45]:


mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)   


# ### X and Y coordinates are obtained from the multi-dimensional scaling model for plotting

# In[46]:


xs, ys = pos[:, 0], pos[:, 1]


# ### Multi dimensional scaling scatter plot

# In[47]:


fig = plt.figure(figsize=(6, 4))
for x, y, page in zip(xs, ys, pages): #pags are page numbers that have been extracted from the dictionary
    plt.scatter(x, y)
    plt.text(x, y, page)
plt.savefig('Figure1.jpg', bbox_inches='tight')
    


# ### Dense cluster in the middle of the graph suggested a dense based clustering algorithm approach

# In[48]:


estimator = DBSCAN(eps=0.1, min_samples=3)
estimator.fit(pos)
labels = estimator.labels_


# ### Labels/clusters for each page are counted with the Counter() function

# In[49]:


print (Counter(labels)) #cluster summery


# ### Reducing the number of colors in the graph and eliminating the page numbers allows a better view of the clusters

# In[50]:


def set_colors(labels, colors='rgbykcm'): #labels correspond to the clusters
    colored_labels = []
    for label in labels:
        colored_labels.append(colors[label])
    return colored_labels


# In[51]:


colors = set_colors(labels)
plt.scatter(xs, ys, c=colors)
plt.savefig('Figure2.jpg', bbox_inches='tight')
plt.show()


# In[52]:


df = pd.DataFrame((zip(pages , labels)), columns =['page', 'cluster']) #names and cluster labels are saved in a data frame


# In[53]:


print(len(df)) #checking the length of the data frame


# In[54]:


df['page'] = df['page'].astype(int) #converting page and cluster columns to int (to be able to sort by page)
df['cluster'] = df['cluster'].astype(int)


# In[55]:


df_sort = df.sort_values(by=['page', 'cluster'],kind="mergesort") #sorting the data frame


# ### The seaborn FacetGrid visualization indicates main text as a long continous orange line

# In[56]:


fg = seaborn.FacetGrid(data=df_sort, hue='cluster', aspect=1.61)
fg.map(plt.scatter, 'page', 'cluster').add_legend()
fg.set(xticklabels=[])
fg.set(xlabel='pages')


# ### Minimum and maximum page of the 0 cluster (typically the largest cluster) represents the boundaries of main content

# In[57]:


cluster_0 = df_sort['cluster'] == 0
cluster0 = df_sort[cluster_0]
page_min = cluster0['page'].min()
page_max = cluster0['page'].max()


# In[58]:


print('Main content starts at page' + ' ' + str(page_min) + ' ' + 'and ends at page' + ' ' + str(page_max), '.')


# ### Applying majority vote smoothing to the results by evaluating three labels before and three after each element in the cluster/label column

# In[59]:


cluster = df_sort['cluster'].tolist() #converting data frame columns to list format for easier manipulation
page = df_sort['page'].tolist()


# In[60]:


window = 3
context =[] #context labels/majority voting placeholder
for backward, current in enumerate(range(len(cluster)), start=0-window):
    if backward < 0:
        backward = 0
    forward = cluster[current:current+1+window] #labels for three pages after the current page
    backward = cluster[backward:current] #labels for three pages before the current page
    out = forward + backward
    context.append(out)


# In[61]:


smoothed = []
for list in context:
    i = Counter(list).most_common(1) #majority voting
    smoothed.append(i)
cluster_votes = [tup[0] for tup in smoothed] # indicates the surrounding context for each label/cluster, how many votes each cluster/label received

cluster_s =[]
for i in cluster_votes:
    cluster_s.append(i[0]) #results after applying majority voting based on context


# In[62]:


cluster_smooth = pd.DataFrame(zip(page, cluster_s), columns = ['page', 'cluster_smooth'])


# In[63]:


cluster_0 = cluster_smooth['cluster_smooth'] == 0
cluster_s_0 = cluster_smooth[cluster_0]
page_min = cluster_s_0['page'].min()
page_max = cluster_s_0['page'].max()


# ### Updated results with smoothing

# In[64]:


print('Main content starts at page' + ' ' + str(page_min) + ' ' + 'and ends at page' + ' ' + str(page_max) + '.')


# In[ ]:





# In[ ]:




