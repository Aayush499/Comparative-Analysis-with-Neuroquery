from scipy import io
import numpy as np
import pandas as pd

story_features_df = pd.read_csv('story_features.csv')
story_features_df = story_features_df.drop('Unnamed: 0', axis=1)

subject_1 = io.loadmat('subject_1.mat')

text = []
start = []
length = []

TIME = pd.DataFrame(subject_1['time'], columns=['onset', 'block'])

for i in np.arange(5176):
    text.append(subject_1['words']['text'][0][i][0][0][0])
    start.append(subject_1['words']['start'][0][i][0][0])
    length.append(subject_1['words']['length'][0][i][0][0])

events = pd.DataFrame({'trial_type': text, 'onset': start, 'duration': length})

new_df = []

for i in np.arange(0, 5176, 4):
    this = events['trial_type'][i:]
    separator=" "
    this_col = separator.join(this[:4])
    this_story = story_features_df[i:]
    su = this_story[:4].sum(axis=0)
    this_su_df = pd.DataFrame(su, columns=[this_col])
    new_df.append(this_su_df)
new_df = pd.concat(new_df, axis=1)
new_df = new_df.reset_index(drop=True)

all_data = []
for feature_idx in np.arange(new_df.shape[0]):
    test = new_df.iloc[feature_idx].reset_index()
    test = test.rename(columns={test.columns[1]: 'feature'})

    new_data = {}

    for i in [1, 2, 3, 4]:
        this_block = TIME[TIME['block'] == i]
        if i == 1:
            this_block_number = this_block.shape[0]
            this_range = np.arange(this_block_number)
            data_indices = this_range
        elif i == 2:
            previous_block_number = this_block_number
            this_block_number = this_block_number + this_block.shape[0]
            this_range = np.arange(previous_block_number, this_block_number)
            data_indices = this_range - 14
        elif i == 3:
            previous_block_number = this_block_number
            this_block_number = this_block_number + this_block.shape[0]
            this_range = np.arange(previous_block_number, this_block_number)
            data_indices = this_range - 28
        elif i == 4:
            previous_block_number = this_block_number
            this_block_number = this_block_number + this_block.shape[0]
            this_range = np.arange(previous_block_number, this_block_number)
            data_indices = this_range - 42
        thr_1 = this_range[11]
        thr_2 = this_range[-1] - 3
        for ii in this_range:
            if ii < thr_1:
                new_data['Fixation_' + str(ii)] = 0.0
            elif ii >= thr_1 and ii <= thr_2:
                this_idx = data_indices[ii - thr_1]
                if this_idx != 1294:
                    this_idx_df = test.iloc[this_idx]
                    new_data[this_idx_df['index']] = this_idx_df['feature']
                else:
                    new_data['Fixation_' + str(ii)] = 0.0
            elif ii > thr_2:
                new_data['Fixation_' + str(ii)] = 0.0
    all_data.append(pd.DataFrame(new_data, index=[feature_idx]))
story_features = pd.concat(all_data)

# semantics = story_features.iloc[0:100]
# speech = all_data.iloc[100:102]
# motion = all_data.iloc[102:109]
# emotion = all_data.iloc[109:132]
# verbs = all_data.iloc[132:137]
# characters = all_data.iloc[137:147]
# visual = all_data.iloc[147:149]
# partofspeech = all_data.iloc[149:178]
# dependency = all_data.iloc[178:]

#features = {'semantics': semantics,
#            'speech': speech,
#            'motion': motion,
#            'emotion': emotion,
#            'verbs': verbs,
#            'characters': characters,
#            'visual': visual,
#            'partofspeech': partofspeech,
#            'dependency': dependency}

subject_1 = io.loadmat('subject_1.mat')
data = subject_1['data']
voxel_1 = data[:, 0]

total_features_df = {}

for t in range(story_features.shape[0]):
    all_features = {}
    for j in range(story_features.shape[1]):
        this_feature = []
        this_j = story_features.iloc[:, j]
        for k in np.arange(1, 5):
            this_k = this_j.iloc[t - k]
            this_feature.append(this_k)
        all_features[j] = this_feature
    total_features_df[t] = pd.DataFrame(all_features)

k = {}
total = []
for i in total_features_df:
    k[str(i) + '_1'] = total_features_df[i].iloc[0]
    k[str(i) + '_2'] = total_features_df[i].iloc[1]
    k[str(i) + '_3'] = total_features_df[i].iloc[2]
    k[str(i) + '_4'] = total_features_df[i].iloc[3]
    total.append(k)

modeling_data = pd.DataFrame(total[0])

from sklearn.linear_model import RidgeCV
coefs = []
for each_feature in range(story_features.shape[0]):
    this_y = modeling_data[[str(each_feature) + '_1', str(each_feature) + '_2',
                            str(each_feature) + '_3', str(each_feature) + '_4']]
    ridge = RidgeCV(alphas=np.logspace(-10, 10, 21))
    ridge.fit(voxel_1.reshape(-1, 1), this_y)
    this_coef = pd.DataFrame(ridge.coef_.squeeze(), columns=[each_feature])
    coefs.append(this_coef)
