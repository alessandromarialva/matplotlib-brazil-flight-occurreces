
# coding: utf-8

# # Analyzing Aviation Occurrences in Brazil
# 
# In this session, we will use a Notebook running Python 3.5 with Apache Spark 2.1 for data analysis and visualizations using Apache SystemML and pandas DataFrames. We'll walk through some examples of Data Preparation, Data Analysis, Scoring, Classification, Data Normalization, Correlations, Feature scaling and normalization.
# 
# We will analyze open data from Opendata AIG Brazil - Brazilian Civil Aviation Occurrences and to extract even more insights, we will explore two datasets and build charts for  visualization of specific areas and see how the data science can help predicting occurrences.
# 
# Term “Occurrence” is defined as “accident or incident” throughout this data analysis.
# 

# # Install prerequisites
# 
# To start, we will import some basic libraries including Machine Learning entirely and import some functions

# In[64]:


#Install SystemML
get_ipython().system(u'pip install --upgrade systemml')


# In[65]:


#Import NumPy and pandas
import numpy as np
import pandas as pd


# # This dataset
# 
# We'll use two datasets, one with all the details of aircrafts involved on every brazilian civil aviation occurrence and other with details of each occurrence in 10 years (2008-2017) of Brazilian Civil Aviation:
# 
# * aircrafts.csv - All the aircrafts that were involved on every occurrence
# 
# * occurrences.csv - All occurences details from brazilian civil aviation
# 
# 
# These two datasets and more can be found at:
# 
# ## Opendata AIG Brazil - Occurrences in Brazilian Civil Aviation
# 
# CENIPA (Centro de Investigação e Prevenção de Acidentes Aeronáuticos)
# 
# http://dados.gov.br/dataset/ocorrencias-aeronauticas-da-aviacao-civil-brasileira
# 

# # Import the 'occurrences.csv' dataset into the Notebook
# 
# Before we load both files into your IBM Cloud Object Storage by dragging and dropping the files on the '1001' pannel, you need to create a connection in your Notebook to your IBM Cloud Object Storage. To do that, from your project page, click on the 'Add to Project' and then on 'Connection', choose your Cloudant Instance and then 'Create'.
# 
# And then we will connect the Notebook to the IBM Cloud Object Storage and load the "aircrafts.csv" dataset to start with our data analysis.

# In[66]:


# Connect your Object Storage by clicking 'Insert Credentials' from the drop-down of the "aircrafts.csv" dataset
# @hidden_cell
# The following code contains the credentials for a file in your IBM Cloud Object Storage.
# You might want to remove those credentials before you share your notebook.
credentials_1 = {
    'IBM_API_KEY_ID': '********',
    'IAM_SERVICE_ID': '********',
    'ENDPOINT': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
    'IBM_AUTH_ENDPOINT': 'https://iam.bluemix.net/oidc/token',
    'BUCKET': 'default-donotdelete-pr-nlfssruajbx9xv',
    'FILE': 'aircrafts.csv'
}


# Import ibm_boto3 library which provides complete access to the IBM Cloud Object Storage API and Insert pandas DataFrame to start exploring our first dataset

# In[67]:


# Import ibm_boto3 library
import types
from ibm_botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_689c19f611b0478583c74d2a7431addc = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='******',
    ibm_auth_endpoint="******",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3-api.us-geo.objectstorage.service.networklayer.com')


# Insert your pandas DataFrame here by clicking on 'Insert pandas DataFrame' from the drop-down of the "aircrafts.csv" dataset 
body = client_689c19f611b0478583c74d2a7431addc.get_object(Bucket='default-donotdelete-pr-nlfssruajbx9xv',Key='aircrafts.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_1 = pd.read_csv(body, na_values=['****', '***',''], header=0, encoding='latin-1')
df_data_1.head()


# # Data Preparation
# 
# Now we will be the accessing, organizing, and structuring of unprocessed data assets to be used for data analysis.
# 
# Notice that some column names contain spaces, we will replace all spaces with underlines.

# In[68]:


# Replace 
df_data_1.columns = [c.replace(' ', '_') for c in df_data_1.columns]
df_data_1.head()


# # Initial Data Exploration with matplotlib
# 
# Import matplotlib, define your own style and plot the results for a better data visualization. 

# In[69]:


# Plotting with matplotlib
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.plotly as py
from plotly.graph_objs import *
pyo.offline.init_notebook_mode()
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = 30

# Defining your own style
plt.style.use('ggplot')
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['lines.linewidth'] = 7
plt.rcParams['lines.markersize'] = 14
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 18


# # Data Analysis
# 
# As an example, we will check the count of the aircrafts that had any sort of failure, grouping by model and engine_type, we will be able to plot the top 20 failed aircrafts.

# In[70]:


FlightsOccur_Sorted = pd.DataFrame(df_data_1.groupby(by=['model','engine_type'])['equipment'].count().reset_index())
FlightsOccur_Sorted.columns = ['model','engine_type','Count']
FlightsOccur_Sorted = FlightsOccur_Sorted.sort_values(by='Count',ascending=False)
data=[]
FlightsOccur_Sorted_Top=FlightsOccur_Sorted.iloc[:20,:]
for Airname in list(FlightsOccur_Sorted_Top['engine_type'].unique()):
    data.append(go.Bar(
    x = list(FlightsOccur_Sorted_Top[FlightsOccur_Sorted_Top['engine_type']==Airname]['model']),
    y = list(FlightsOccur_Sorted_Top[FlightsOccur_Sorted_Top['engine_type']==Airname]['Count']),
    name= Airname
    )  
    )
layout = go.Layout(
    barmode='group',
    title = 'Top 20 failed Aircrafts'
)
fig = go.Figure(data=data, layout=layout)
pyo.iplot(fig)


# Clearly we can see that model 'AB-115' that uses PISTON engines was the top failed Aircraft, followed by models 'EMB-202', 'EMB-201A' and others as you can see above.

# # Scoring
# 
# In machine learning, also called Prediction, Scoring is the process of applying an algorithmic model built from a historical dataset to a new dataset in order to uncover practical insights that will help solving problems.
# 
# Taking as an example, the damage_level column by having the values UNKNOWN, NONE, LIGHT, SUBSTANTIAL and DESTROYED. We will score the values to nan, 0, 1, 2, 3 respectively.
# 

# In[71]:


# Scoring
df_data_1.replace(to_replace=['UNKNOWN', 'NONE', 'LIGHT', 'SUBSTANTIAL', 'DESTROYED'], value=[np.nan, 0, 1, 2, 3], inplace=True)
df_data_1.head()


# We will also drop the rows where both 'damage_level' and 'engines_amount' columns have the NaN value, excluding a few rows of our data.
# 

# In[72]:


# Drop all rows where 'damage_level' & 'engines_amount' columns have NaN value
df_data_1.dropna(axis=0, how='any', subset=['damage_level', 'engines_amount'], inplace=True)
df_data_1.head()


# Let's explore a little more this Dataset and see the top 10 failed aircrafts based on the operation phases of the Occurrence to which Aircrafts and in which operation phase failed.

# In[73]:


FlightsOccur_Sorted = pd.DataFrame(df_data_1.groupby(by=['model','operation_phase'])['equipment'].count().reset_index())
FlightsOccur_Sorted.columns = ['model','operation_phase','Count']
FlightsOccur_Sorted = FlightsOccur_Sorted.sort_values(by='Count',ascending=False)
data=[]
FlightsOccur_Sorted_Top=FlightsOccur_Sorted.iloc[:10,:]
for Airname in list(FlightsOccur_Sorted_Top['operation_phase'].unique()):
    data.append(go.Bar(
    x = list(FlightsOccur_Sorted_Top[FlightsOccur_Sorted_Top['operation_phase']==Airname]['model']),
    y = list(FlightsOccur_Sorted_Top[FlightsOccur_Sorted_Top['operation_phase']==Airname]['Count']),
    name= Airname
    )  
    )
layout = go.Layout(
    barmode='group',
    title = 'Top aircrafts fails based on the operation phases of the Occurrence'
)
fig = go.Figure(data=data, layout=layout)
pyo.iplot(fig)


# Curiously, only two of all aircrafts had occurrences during the 'SPECIALISED' operation phase.
# 
# Specifically, the 'AB-115' model had 34 Landing, 24 Takeoff and 15 Run After Landing occurrences.

# # Import the 'occurrences.csv' dataset
# 
# We will import the "occurrences.csv" dataset into the Notebook and explore our second dataset.
# 

# In[74]:


#Insert your pandas DataFrame by clicking 'Insert pandas DataFrame' from the drop-down of the "occurrences.csv" dataset
body = client_689c19f611b0478583c74d2a7431addc.get_object(Bucket='default-donotdelete-pr-nlfssruajbx9xv',Key='occurrences.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_2 = pd.read_csv(body, na_values=['****', '***',''], header=0, encoding='latin-1')
df_data_2.head()


# # Data preparation
# 
# Notice that some column names contain spaces, we will also replace every space with an underline.
# 

# In[75]:


# Replace
df_data_2.columns = [c.replace(' ', '_') for c in df_data_2.columns]
df_data_2.head()


# As an example, we will create a new column called 'occurrence_year' and extract the year from 'occurrence_day' column, which is composed of Month, Day and Year. 

# In[76]:


# Create a new column called 'occurrence_year' based on the 'occurrence_day' column
df_data_2['occurrence_year'] = pd.to_datetime(df_data_2['occurrence_day'], format='%m/%d/%Y').dt.year
df_data_2.head()


# # Classification
# 
# Classification can be performed on structured or unstructured data. Classification is a technique where we categorize data into a given number of classes. The main goal of a classification problem is to identify the category/class to which a new data will fall under.
# 

# In[77]:


# Classification
print(df_data_2['classification'].unique())


# Now we will get the count of 'ACCIDENT' and 'SERIOUS INCIDENT' and also the total count

# In[78]:


# Count
print('Total: ' + str(df_data_2['classification'].count()))
print('SERIOUS INCIDENT: ' + str(df_data_2['classification'][df_data_2['classification'] == 'SERIOUS INCIDENT'].count()))
print('ACCIDENT: ' + str(df_data_2['classification'][df_data_2['classification'] == 'ACCIDENT'].count()))


# List all Occurrece Types

# In[79]:


# List all occurrence types
print(df_data_2['type_of_occurrence'].unique())


# Also get the count of each occurrence type

# In[80]:


# Count of each occurrence type
for occ in df_data_2['type_of_occurrence'].unique():
    print(occ + ': ' + str(df_data_2['type_of_occurrence'][df_data_2['type_of_occurrence'] == occ].count()))


# Plot the results for a better view

# In[81]:


FlightOccur_Sorted = pd.DataFrame(df_data_2.groupby(by=['type_of_occurrence','classification'])['fu'].count().reset_index())
FlightOccur_Sorted.columns = ['type_of_occurrence','classification','Count']
FlightOccur_Sorted = FlightOccur_Sorted.sort_values(by='Count',ascending=False)
data=[]
FlightOccur_Sorted_Top=FlightOccur_Sorted.iloc[:10,:]
for Airname in list(FlightOccur_Sorted_Top['classification'].unique()):
    data.append(go.Bar(
    x = list(FlightOccur_Sorted_Top[FlightOccur_Sorted_Top['classification']==Airname]['type_of_occurrence']),
    y = list(FlightOccur_Sorted_Top[FlightOccur_Sorted_Top['classification']==Airname]['Count']),
    name= Airname
    )  
    )
layout = go.Layout(
    barmode='group',
    title = 'Top 10 aircrafts fails based on the operation phases of the Occurrence'
)
fig = go.Figure(data=data, layout=layout)
pyo.iplot(fig)


# Engine Failure during the Flight is the top occurrence type followed by Loss of Control in The Air as you can see above.

# # Applying Data Normalization to the Occurrence Type field
# 
# Data normalization means transforming all variables in the data to a specific range. We do data normalization when seeking for relations.

# In[82]:


# Data Normalization
occurrences_freq_dict = {}
for occ in df_data_2['type_of_occurrence'].unique():occurrences_freq_dict[occ] = df_data_2['type_of_occurrence'][df_data_2['type_of_occurrence'] == occ].count() / df_data_2['type_of_occurrence'].count()
occurrences_freq = pd.Series(occurrences_freq_dict)
occurrences_freq.head(10)


# # Now let's see the count for each State
# Notice the "EX" isn't a state and it relates to accurrences happened out side of Brazilian territory

# In[83]:


for occ in df_data_2['fu'].unique():
    print(str(occ) + ' :' + str(df_data_2['fu'][df_data_2['fu'] == occ].count()))


# # Applying Data Normalization for the States field
# 
# Database normalization is the process of structuring a relational database in accordance with a series of so-called normal forms in order to reduce data redundancy and improve data integrity.

# In[84]:


states_freq_dict = {}
for occ in df_data_2['fu'].unique():
    states_freq_dict[occ] = df_data_2['fu'][df_data_2['fu'] == occ].count() / df_data_2['fu'].count()
states_freq = pd.Series(states_freq_dict)
states_freq.head(30)


# # Plotting the results of Flight Occurrence rate
# 
# Now we will plot the rate results

# In[85]:


fig, axes = plt.subplots(figsize=(20.,6.))
states_freq.sort_values(ascending=False).plot(kind='bar')

axes.set_xlabel('State')
axes.set_ylabel('Flight Occurrence rate')
                   
plt.show()
plt.close()


# Clearly we see 'SP' state as the top followed by 'RS', 'PR', 'MT', 'MG' and other as shown above.

# # Flights evolution per state from 2006 to 2015
# 
# The results will show the rate for each state, also look for the 'All' column to have a better idea of the evolution 

# In[86]:


freq_year_dict = { }
freq_year_dict['all'] = {}

for s in df_data_2['fu'].unique():freq_year_dict[s] = {}

for occ in df_data_2['occurrence_year'].unique():
    
    freq_year_dict['all'][occ] = df_data_2['occurrence_year'][df_data_2['occurrence_year'] == occ].count()

    for s in df_data_2['fu'].unique():
        
        freq_year_dict[s][occ] = df_data_2['occurrence_year'][(df_data_2['occurrence_year'] == occ) &(df_data_2['fu'] == s)].count()

freq_year = pd.DataFrame(freq_year_dict)

freq_year.head(10)


# # Correlations
# 
# Correlation measure how two observed variables are related to each other. It has been used in many different ways in data science.
# 
# Good candidates for "features" determining the value of damage_level are:
# 
# * engines_amount
# * seatings_amount
# * takeoff_max_weight_(Lbs)
# * year_manufacture
# 
# Let's investigate how they correlate to damage_level.

# In[87]:


column_list = ['damage_level', 'engines_amount', 'seatings_amount', 'takeoff_max_weight_(Lbs)', 'year_manufacture']

fig, axes = plt.subplots(nrows=4, sharex=True, figsize=(20.,20.))
plt.subplots_adjust(hspace=0.4)

for j in range(1, len(column_list)):
  
    axes[j-1].scatter(df_data_1[column_list[0]], df_data_1[column_list[j]], color='black', alpha=0.5)
    
    axes[j-1].set_ylabel(column_list[j])

axes[0].set_xlim([-1.,4.])
axes[3].set_ylim([1935, 2016])

axes[3].set_xlabel(column_list[0])

plt.show()
plt.close()


# The only correlation visible is the negative correlation between takeoff_max_weight and damage_level.
# 
# To really understand correlations, we need to compute the Pearson correlation coefficients and build a full correlation matrix.

# In[88]:


def plot_heatmap(df):
    import seaborn as sns
    fig, axes = plt.subplots()
    sns.heatmap(df, annot=True)
    plt.xticks(rotation=90)
    
    plt.show()
    plt.close()
    
plot_heatmap(df_data_1[column_list].corr(method='pearson'))


# # Parallel coordinates plot
# 
# "Parallel coordinates is a plotting technique for plotting multivariate data. It allows one to see clusters in data and to estimate other statistics visually. Using parallel coordinates points are represented as connected line segments. Each vertical line represents one attribute. One set of connected line segments represents one data point. Points that tend to cluster will appear closer together".

# In[89]:


# Feature scaling and normalization
from pandas.plotting import parallel_coordinates
def z_score_norm(df, feat):
    
    dff = df.copy(deep=True)
    
    dff[feat] = (df[feat] - df[feat].mean()) / (df[feat].max() - df[feat].min())
    
    return dff

aircrafts_norm = z_score_norm(df_data_1[column_list], column_list[1:])

fig, axes = plt.subplots(figsize=(10.,6.))

parallel_coordinates(aircrafts_norm, 'damage_level')

plt.show()
plt.close()


# #  Andrews plot
# 
# "Andrews curves allow one to plot multivariate data as a large number of curves that are created using the attributes of samples as coefficients for Fourier series. By coloring these curves differently for each class it is possible to visualize data clustering. Curves belonging to samples of the same class will usually be closer together and form larger structures".

# In[90]:


from pandas.plotting import andrews_curves
andrews_curves(aircrafts_norm, 'damage_level')


# That's an awesome plot: while in the parallel coordinates plot it was impossible to see any pattern, here we verify that 1 ('NONE') and 3 ('DESTROYED') are clustered at some level almost "everywhere".

# # Conclusion:
# 
# * Most points are opposite to takeoff_max_weight_(Lbs) and on the engines_amount side.
# 
# * Most points are opposite to seatings_amount and on the year_manufacture side, but too much spread on this axis.
# 
# Therefore, engines_amount and takeoff_max_weight_(Lbs) seems to be the most reliable features to characterize damage_level.
