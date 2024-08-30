# python2024 - behavioural data analyses and plot
Project for the python course 2024

The aim of this project is to perform analyses and plot data from a behavioural experiment. 

## Experimental design

The experiment is a spatial ventriloquis experiment.

* The auditory and visual stimuli were sampled separately from **three different positions** (left, center and right), and could either be congruent or incongruent, resulting in **3 levels of spatial disparity** (None: 0° of visual angle, Low: 9° of visual angle, High: 18° of visual angle).
* Stimuli could be **communicative** (video of a person uttering a word) or **non-communicative** (static frame + vocalization)
* Participants had to **localise the visual stimuli or the acoustic stimuli** according to a cue presented after the offset of the stimuli.

These factors result in a 3 (AV disparity) x 2 (communicativeness) x 2 (task modality) factorial design, with a total of 12 conditions. 

## Data

1. Participants' average and median **Response Times** for each condition are stored in a .csv file *'rt_direct.csv'*, with 34 rows (number of participants) and 25 columns (participant id + conditions average and median).
2. Participants' average **Audiovisual Weight Indexes** (index of audiovisual integration) for each incongruent condition are stored in a .csv file *'subj_wav_direct.csv'*, with 34 rows (participants) and 8 columns (number of incongruent conditions).
3. Group **response distributions** are stored in a 3D matrix .mat *'hist_resp_group_mean.mat'*, which is a 3D 9x3x4 file containing, for each of the 4 CommxTask combinations, the percentage of participants responses for each of the three possible positions (left/center/right), in each of the 9 AV spatial combination (3x3).

## Analyses and Plots

1. Response Time repeated measures ANOVA
2. Distribution and box plot of Response Times for each condition
3. Line plor of group Mean and SEM of the Audiovisual Weight Index for each of the incongruent condition
4. Correlation plots for each of the incongruent conditions to show the possible relationship between response times and Audiovisual Weight Indexes
5. Distribution plots of participants responses

## Libraries

Libraries | Documentation
----------|--------------
numpy | https://numpy.org/doc/
pandas | https://pandas.pydata.org/docs/index.html
matplotlib | https://matplotlib.org/stable/index.html
statsmodels | https://statsmodels.org/stable/index.html
seaborn | https://seaborn.pydata.org/
brokenaxes | https://test-brokenaxes.readthedocs.io/en/latest/
scipy | https://docs.scipy.org/doc/scipy/reference/io.html


