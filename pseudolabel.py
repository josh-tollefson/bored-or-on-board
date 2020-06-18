import pandas as pd
import wordprocess as wp
import utils

### Pseudo-labeling for categorization of comments ------- ###
### ------------------------------------------------------ ###
### User inputs ------------------------------------------ ###
### Words in sentences may be changed to your best opinion ###

NCOMMENTS = 10
FOLDER = './save/'
INPUT_PANDAS = 'comments_regression.jl'
OUTPUT_PANDAS = 'bgg_psuedolabels_ncomments' + str(NCOMMENTS) + '.csv'

mechanics_sentence = "competitive interaction strategy tactics replay gameplay mechanic system elimination random decision tense deduction luck scoring dynamics"
theme_sentence = 'art theme beautiful colorful design atmospheric universe'
component_sentence = 'component layout marker tile box small large wear insert worn miniature minis production'
length_sentence = 'slow quick fast filler setup takedown hours long'
accessible_sentence = 'rule family learn easy hard difficult difficulty disability weight complex complicated accessible simple'

##############################################################

df = utils.load_dataframe(FOLDER, INPUT_PANDAS)
df = df.iloc[:NCOMMENTS]

dict_bg_tags_process = {'mechanics': wp.preprocessor(mechanics_sentence, labels=True), \
						'theme': wp.preprocessor(theme_sentence, labels=True), \
						'component': wp.preprocessor(component_sentence, labels=True), \
						'length': wp.preprocessor(length_sentence, labels=True), \
						'accessible': wp.preprocessor(accessible_sentence, labels=True)}

df['mechanics'] = pd.DataFrame({'mechanics': np.zeros((df.shape[0]))})
df['theme'] = pd.DataFrame({'theme': np.zeros((df.shape[0]))})
df['component'] = pd.DataFrame({'component': np.zeros((df.shape[0]))})
df['length'] = pd.DataFrame({'length': np.zeros((df.shape[0]))})
df['accessibility'] = pd.DataFrame({'accessibility': np.zeros((df.shape[0]))})

for i in range(df.shape[0]):
	tokens = wp.preprocessor(df['comment'].loc[i], labels=True)

	mechanics_found = False
	theme_found = False
	component_found = False
	length_found = False
	accessibility_found = False

	for token in tokens:

		if token in dict_bg_tags_process['mechanics']:
			df['mechanics'].loc[i] = 1
			mechanics_found = True

		if token in dict_bg_tags_process['theme']:
			df['theme'].loc[i] = 1
			theme_found = True

		if token in dict_bg_tags_process['component']:
			df['component'].loc[i] = 1
			component_found = True

		if token in dict_bg_tags_process['length']:
			df['length'].loc[i] = 1
			length_found = True

		if token in dict_bg_tags_process['accessible']:
			df['accessibility'].loc[i] = 1
			accessibility_found = True

	if i % 100 == 0:
		print('On Row number: %d ' % i)

utils.save_dataframe(df, FOLDER, OUTPUT_PANDAS)


