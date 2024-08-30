#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%matplotlib widget

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.stats.anova import AnovaRM 
import seaborn as sns
from brokenaxes import brokenaxes
import scipy.io

plt.rc('axes.spines', bottom=True, left=True, right=False, top=False)


# # Response times repeated measures ANOVA
# The RT .csv is a 34x25 file containing average (A) and meadian (M) response times for each participant (rows) in each of the 12 conditions (columns).
# Conditions are according to a 2 (Communicative vs. Non-Communicative) x 2 (auditory vs. visual task) x 3 (levels of audiovisual disparity: no disparity/0°, low disparity/9°, high disparity/18°) factorial space.
# 
# In the following code we take the RT dataframe from a wide to a long format, where each line is a trial, to then perform repeated measures ANOVA using the statsmodels module.

# In[2]:


# read csv file containing participants median and average RT for each condition (3x2x2)
rt_df = pd.read_csv('rt_direct.csv')

# only keep median columns
rt_df = rt_df.drop(columns=['ACA0', 'ACA1', 'ACA2', 'ACV0', 'ACV1', 'ACV2', 'ANA0', 'ANA1', 'ANA2', 'ANV0', 'ANV1', 'ANV2'])
rt_df.rename(columns={'MCA0': 'C_A_0', 'MCA1': 'C_A_9', 'MCA2': 'C_A_18', 'MCV0': 'C_V_0', 'MCV1': 'C_V_9', 'MCV2': 'C_V_18',
                     'MNA0': 'NC_A_0', 'MNA1': 'NC_A_9', 'MNA2': 'NC_A_18', 'MNV0': 'NC_V_0', 'MNV1': 'NC_V_9', 'MNV2': 'NC_V_18'}, inplace=True)

# transform the wide df in a long df, with a row for each condition x participant
rt_df_long = pd.melt(rt_df, id_vars=['Participant'], var_name='Condition', value_name='ResponseTime')

# split the conditions into three separate factors
rt_df_long[['Action', 'Task', 'AV_disparity']] = rt_df_long['Condition'].str.split('_', expand=True)

# sort by participant and reset indexes
rt_df_long = rt_df_long.sort_values(by = 'Participant')
rt_df_long = rt_df_long.reset_index(drop = 'true')


# In[3]:


# repeated measures ANOVA to evaluate the effect of communicativeness, task relevance and AV disparity on participants'
# response times in the spatial localisation task
rt_anova = AnovaRM(rt_df_long, 'ResponseTime', 'Participant', within=['Action', 'Task', 'AV_disparity']).fit()
print(rt_anova)


# In[4]:


# group means and sem for each condition
rt_df_gm = rt_df_long.groupby(['Action', 'Task', 'AV_disparity'])['ResponseTime'].mean()
rt_df_sem = rt_df_long.groupby(['Action', 'Task', 'AV_disparity'])['ResponseTime'].sem()


# # Response Times plot
# We plot the distribution and mean Response Times for each condition. The auditory task is on the left blue plot, with light color for communicative conditions and dark color for non-communicative conditions; the visual task is on the right green plot. Response time are represented in the y axis and AV disparity cotegories in the x axis.

# In[5]:


# set subplots and axes options
fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
axes[0].set(title= 'Auditory Task', ylabel="Response Time (ms)", xticks=[0,1,2,3,4,5], xticklabels=['0', '9', '18', '0', '9', '18'])
axes[1].set(title= 'Visual Task', xticks=[0,1,2,3,4,5], xticklabels=['0', '9', '18', '0', '9', '18'])
axes[0].tick_params(axis='x', which='both', length=0)
axes[1].tick_params(axis='both', which='both', length=0)

# plot auditory task responses in left side plot, separating actions by color and AV disparity in different groups
sns.swarmplot(rt_df[['C_A_0', 'C_A_9', 'C_A_18']], ax=axes[0], orient='v', color='#a6cee3')
sns.swarmplot(rt_df[['NC_A_0', 'NC_A_9', 'NC_A_18']], ax=axes[0], orient='v', color='#1f78b4')
sns.boxplot(rt_df[['C_A_0', 'C_A_9', 'C_A_18', 'NC_A_0', 'NC_A_9', 'NC_A_18']], ax=axes[0], orient='v', boxprops={'facecolor':'none'})

# plot visual task responses in right side plot
sns.swarmplot(rt_df[['C_V_0', 'C_V_9', 'C_V_18']], ax=axes[1], orient='v', color='#b2df8a')
sns.swarmplot(rt_df[['NC_V_0', 'NC_V_9', 'NC_V_18']], ax=axes[1], orient='v', color='#33a02c')
sns.boxplot(rt_df[['C_V_0', 'C_V_9', 'C_V_18', 'NC_V_0', 'NC_V_9', 'NC_V_18']], ax=axes[1], orient='v', boxprops={'facecolor':'none'})
#sns.despine(ax=axes[0], top=True, right=True)
sns.despine(ax=axes[1], left=True)

# set legends
# left plot
handles0 = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#a6cee3', markersize=10, linestyle=''),
           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f78b4', markersize=10, linestyle='')]
labels0 = ['Communicative', 'Non-communicative']
axes[0].legend(handles=handles0, labels=labels0, loc='upper left')
# right plot
handles1 = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#b2df8a', markersize=10, linestyle=''),
           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#33a02c', markersize=10, linestyle='')]
labels1 = ['Communicative', 'Non-communicative']
axes[1].legend(handles=handles1, labels=labels1, loc='upper left')

# common x axis label
fig.text(0.5, 0.01, 'Audiovisual spatial disparity (° of visual angle)', ha='center', fontsize=10)
plt.tight_layout(rect=[0, 0.03, 1, 1])

plt.show()


# # Audiovisual weight index (Wav) plot
# The audiovisual weight index (Wav) is an index of multisensory integration, with values between 0 and 1. Values closer to 0 indicate a greater influence of the acoustic stimuli, while values closer to 1 indicate a greater influence of the visual stimuli.
# 
# The .csv is a 34x8 file containing mean Wav values for each participant (rows) in each of the conditions. The index can be calculated only for incongruent conditions (AV disparity > 0°) so the conditions are in a 2 (Communicative/Non-communciative) x 2 (Auditory/Visual Task) x 2 (low/high AV disparity).
# 
# We plot the group mean and SEM on the y axis, and the comm/non-communicative condition on the x axis. Task modality and AV disparity are represented by differen lines.
# 
# 

# In[6]:


# read audiovisual weight indexes for 34 participants (rows) in 8 spatially incongruent conditions 
s_wav = pd.read_csv('subj_wav_direct.csv')

# create dictionaries with average and sem per condition
mean_wav = s_wav.mean()
mean_wav = mean_wav.to_dict()
sem_wav = s_wav.sem()
sem_wav = sem_wav.to_dict()

# lines to plot
low_aud = ['mean_low_com_aud','mean_low_nc_aud']
high_aud = ['mean_high_com_aud','mean_high_nc_aud']
low_vis = ['mean_low_com_vis','mean_low_nc_vis']
high_vis = ['mean_high_com_vis','mean_high_nc_vis']


# x and y values for the plot for means and SEM
x_cat = ['Comm','Non-comm']
low_aud_y = [mean_wav[key] for key in low_aud]
high_aud_y = [mean_wav[key] for key in high_aud]
low_vis_y = [mean_wav[key] for key in low_vis]
high_vis_y = [mean_wav[key] for key in high_vis]

low_aud_sem = [sem_wav[key] for key in low_aud]
high_aud_sem = [sem_wav[key] for key in high_aud]
low_vis_sem = [sem_wav[key] for key in low_vis]
high_vis_sem = [sem_wav[key] for key in high_vis]

fig = plt.figure(figsize=(3,5))

# break y axis
bax = brokenaxes(ylims=((0, 0.03), (0.38, 1)), hspace=.05)

# plot lines
bax.errorbar(x_cat, low_aud_y, yerr=low_aud_sem, fmt='-o', capsize=5, color='#a6cee3', label='low aud')
bax.errorbar(x_cat, high_aud_y, yerr=high_aud_sem, fmt='-o', capsize=5, color='#1f78b4', label='high aud')
bax.errorbar(x_cat, low_vis_y, yerr=low_vis_sem, fmt='-o', capsize=5, color='#b2df8a', label='low vis')
bax.errorbar(x_cat, high_vis_y, yerr=high_vis_sem, fmt='-o', capsize=5, color='#33a02c', label='high vis')

# add legend
bax.legend(loc='upper left', fontsize=8, bbox_to_anchor=(1.01, 1))

# set x axis ticks
bax.set_xlim(-0.5,1.5)
bax.axs[1].set_xticks([0,1])
bax.axs[1].xaxis.set_tick_params(length=0)
bax.axs[1].set_yticks([0], ['A influence  0'])
bax.axs[0].set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], ['0.4', '0.5', '0.6', '0.7', '0.8', '0.9','V influence  1'])

bax.axs[1].spines['bottom'].set_visible(False)

plt.show()


# # Response times - Wav correlation
# To evaluate a possible correlation between response times and audiovisual weight indexes we plot for each participant, for each incongruent condition, mean RT on the y axis and mean Wav on the x axis. We also fit a nonlinear regression curve to the data.

# In[7]:


# rename columns to match rt dataframe
s_wav.columns = ['C_A_9','NC_A_9','C_A_18','NC_A_18','C_V_9','NC_V_9','C_V_18','NC_V_18']

# plot correlation between integration index and response times for the acoustic task (top row) and visual task (bottom row)
fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)

plot_title = ['Comm-LowDisparity','NonComm-LowDisparity','Comm-HighDisparity','NonComm-HighDisparity']

for index, col in enumerate(s_wav.columns):
    if col in rt_df.columns:

        #find nonlinear fit for the data
        coefficients = np.polyfit(s_wav[col], rt_df[col], deg=2)
        polynomial = np.poly1d(coefficients)

        x_curve = np.linspace(min(s_wav[col]), max(s_wav[col]), 100)
        y_curve = polynomial(x_curve)

        # match the column to the subplot
        if index<4:
            r=0
            c=index
            # xlims and ticks for acoustic task
            xbounds=[0,1.1] 
            xint=[0,0.2,0.4,0.6,0.8,1]
        else:
            r=1
            c=index-4
            # xlims and ticks for visual task
            xbounds=[0.8,1.05]
            xint=[0.8,0.85,0.90,0.95,1]

        # scatter plot
        axes[r][c].scatter(s_wav[col], rt_df[col], alpha=0.7)
        # curve fit
        axes[r][c].plot(x_curve, y_curve, color='red', label="Polynomial Fit (Degree 2)")

        # set axes options
        axes[r][c].set(title = plot_title[c], ylim=(200,1200),xlim=xbounds,xticks=xint)
        axes[r][c].spines['top'].set_visible(False)
        axes[r][c].spines['right'].set_visible(False)

# rows names
fig.text(0.5, 0.94, 'Auditory task', ha='center', va='center', fontsize=15)
fig.text(0.5, 0.49, 'Visual task', ha='center', va='center', fontsize=15)

plt.subplots_adjust(hspace=0.4) #adjust spacing between rows

plt.show()


# # Responses distributions for each AV condition
# The .mat file is a 3D file containing, for each of the 4 CommxTask combinations, the percentage of participants responses for each of the three possible positions (left/center/right), in each of the 9 AV spatial combination (3x3).
# 
# We store each of the 4 CommxTask conditions in a 3x9 dataframe, with an AV spatial condition in each column and the distribution of responses in the rows.
# 
# We then plot the distribution of responses in a 3x3 grid, with each plot representing one AV spatial combinations and different lines for CommxTask conditions (visual task in green, acoustic task in blue, light color for communciative, dark color for non-communicative). The visual stimuli position is represented on the y axis of the x, and the acoustic stimuli position on the x axis.
# Each subplot has the percentage of responses on the y axis and the 3 possible responses (left, center, right) on the x axis.

# In[8]:


# load matlab data
# it's a 3D array --> 9 (AV spatial combinations) x 3 (% of responses for each of the 3 possible responses) x 4 (CommxTask combinations)
data = scipy.io.loadmat('hist_resp_group_mean.mat')
hist = data['hist_mat_group_mean']

# store each dimention in a dataframe
# 0 = C-A
# 1 = NC-V
# 2 = C-V
# 3 = NC-V

list_dfs = []
for i in range(hist.shape[2]):
    df = pd.DataFrame(hist[:,:,i])

    list_dfs.append(df)

# Transpose dataframes for easier plotting
hist_CA = list_dfs[0].T
hist_NCA = list_dfs[1].T
hist_CV = list_dfs[2].T
hist_NCV = list_dfs[3].T


# In[9]:


# 3x3 plot of responses for each CommunicativenessxTask combination
# row 0 = visual left
# row 1 = visual center
# row 2 = visual right
# col 0 = aud left
# col 1 = aud center
# col 2 = aud right

fig, axes = plt.subplots(3,3, figsize=(15,8), sharex=True,sharey=True)
axes = axes.flatten()

# for each of the 9 AV combinations plot the responses for each of the 4 conditions
for col in range(9):
    hist_CA[col].plot(ax=axes[col], kind='line', color='#a6cee3', label='CommunicativeAudTask')
    hist_NCA[col].plot(ax=axes[col], kind='line', color='#1f78b4', label='NonCommunicativeAudTask')
    hist_CV[col].plot(ax=axes[col], kind='line', linestyle='dashed', color='#b2df8a', label='CommunicativeVisTask')
    hist_NCV[col].plot(ax=axes[col], kind='line', linestyle='dashed', color='#33a02c', label='NonCommunicativeVisTask')

    axes[col].set(ylim=(0,1),yticks=[],xlim=(-0.2,2.2),xticks=[])
    axes[col].tick_params(axis='both', which='both', length=0)
    
    
axes[0].set(ylabel='left')
axes[3].set(ylabel='center')
axes[6].set(ylabel='right',xlabel='left')
axes[7].set(xlabel='center')
axes[8].set(xlabel='right')

fig.text(0.5, 0.01, 'Acoustic stimuli position', ha='center', fontsize=14)
fig.text(0.01, 0.5, 'Visual stimuli position', va='center', rotation='vertical', fontsize=14)

axes[2].legend(loc='upper left', bbox_to_anchor=(1.01, 1))

plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])

