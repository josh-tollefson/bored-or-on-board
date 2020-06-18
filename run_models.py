import pandas as pd
import numpy as np
import seaborn as sns
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import wordprocess as wp
from wordprocess import preprocessor
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import streamlit as st
import re
import os

def load_model(folder, infile):

	model = pickle.load( open(folder + infile, 'rb') )

	return model


def load_vectorizer(folder, infile):

	vectorizer = pickle.load( open(folder + infile, 'rb') )

	return vectorizer

@st.cache(suppress_st_warning=True)
def run_sentiment_model(df, modelpath, modelfile, vectorpath, vectorfile):
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.naive_bayes import MultinomialNB

	try:
		comments_test = vectorizer.transform(df['comment'].values)
		Y_pred = model.predict(comments_test)

	except UnboundLocalError:

		model = load_model(modelpath, modelfile)
		vectorizer = load_vectorizer(vectorpath, vectorfile)

		comments_test = vectorizer.transform(df['comment'].values)
		Y_pred = model.predict(comments_test)

	return Y_pred

#@st.cache(suppress_st_warning=True)
def run_label_model(df, modelpath, modelfile, vectorpath, vectorfile, top_n=10):
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.naive_bayes import MultinomialNB

	model = load_model(modelpath, modelfile)
	vectorizer = load_vectorizer(vectorpath, vectorfile)

	docs_test_counts = vectorizer.transform(df['comment'].values).toarray()

	Y_pred = model.predict(docs_test_counts)

	Y_prob = model.predict_proba(docs_test_counts)

	indices = np.argsort(Y_prob[:,1])

	comments_likely = df['comment'].values[indices][-3:]

	dfs = top_feats_by_class(docs_test_counts, Y_pred, np.array(vectorizer.get_feature_names()), top_n=10000)

	features_0 = dfs[0]['feature'].values
	tdif_0 = dfs[0]['tfidf'].values / np.sum(dfs[0]['tfidf'].values)

	features_1 = dfs[1]['feature'].values
	tdif_1 = dfs[1]['tfidf'].values / np.sum(dfs[1]['tfidf'].values)

	unique_features = []
	unique_tdif = []

	for i in range(len(features_1)):
		if features_1[i] not in features_0:

			unique_features.append(features_1[i])
			unique_tdif.append(tdif_1[i]) 
		else:
			ind = np.where(features_0 == features_1[i])
			if tdif_1[i] > 3 * tdif_0[ind]:
				unique_features.append(features_1[i]) 
				unique_tdif.append(tdif_1[i]) 

	for i in range(len(comments_likely)):

		str_0 = wp.preprocessor(comments_likely[i], labels=True)

		for feature in unique_features:

			p = re.compile('\\b'+feature+'\\b')
			result = p.search(str_0) # match
			#print(str_0)
			#print(p)
			if result != None:
				sentences = [sentence for sentence in re.split('[.?!]', comments_likely[i]) if result[0] in sentence]
				try:
					comments_likely[i] = comments_likely[i].replace(sentences[0], "**" + sentences[0].replace('\n','') + "**" )
					
					words = comments_likely[i].partition("**" + sentences[0].replace('\n','') + "**")

					comments_likely[i] = words[1]
					#comments_likely[i] = ' '.join(['...' + words[0].split('.')[-2] + '.', words[1] + '.', words[2].split('.')[1] + '...'])
				except IndexError:
					continue

				break

	return Y_pred, unique_features[:6], unique_tdif[:6], comments_likely

@st.cache(suppress_st_warning=True)
def get_sentiment(ratings, min_rating = 4.0, max_rating = 9.0):
	# Takes dataframe of ratings

	df_sentiment = pd.DataFrame(0, index=np.arange(len(ratings.to_numpy())), columns=['sentiment'])

	df_sentiment['sentiment'].loc[ratings <= min_rating] = 0
	df_sentiment['sentiment'].loc[ratings >= max_rating] = 1
	#df_sentiment['sentiment'].loc[(ratings > min_rating) & (ratings < max_rating)] = np.nan

	return df_sentiment['sentiment']

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''

    from sklearn.feature_selection import chi2

    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids] #if len(features[i].split(' '))== 2]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def top_tfidf_feats_chi2(vectorizer, comments, labels, top_n=25):
    from sklearn.feature_selection import chi2

    features_chi2 = chi2(comments, labels)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(vectorizer.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    trigrams = [v for v in feature_names if len(v.split(' ')) == 3]

    # print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[:top_n])))
    # print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[:top_n])))
    # print("  . Most correlated trigrams:\n. {}".format('\n. '.join(trigrams[:top_n])))
    

    return unigrams[:top_n], bigrams[:top_n]

def top_feats_in_doc(X, features, row_id, top_n=25):
	''' Top tfidf features in specific document (matrix row) '''
	row = np.squeeze(X[row_id].toarray())
	return top_tfidf_feats(row, features, top_n)

def top_mean_feats(X, features, grp_ids=None, min_tfidf=0.001, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = X[grp_ids]
    else:
        D = X

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)

    return top_tfidf_feats(tfidf_means, features, top_n)


def top_feats_by_class(Xtr, y, features, min_tfidf=0.001, top_n=25):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_labels(words, tdif):

	words_df = pd.DataFrame({"words": words, "tdif": tdif})
	sns.barplot(x='words', y='tdif', data=words_df, orient="v")
	del words_df
	st.pyplot()


@st.cache
def load_data(infile):

	df = pd.read_csv(infile)
	df['comment'] = df['comment'].fillna(' ')
	#df['rating'] = get_sentiment(df['rating'])

	return df

def file_selector(folder_path='.'):
   filenames = os.listdir(folder_path)
   selected_filename = st.selectbox('Select a file containing a csv of comments contained in one column, one comment per row', filenames)
   return os.path.join(folder_path, selected_filename)


pd.set_option('mode.chained_assignment', None)


#run_label_model(df, FOLDER, MODELFILE, FOLDER, VECTORFILE, top_n=100)


st.markdown('# Bored, or on Board?')

st.header('1. Data Input')
user_path = st.text_input("Enter path of board game comments", './examples/')

filename = file_selector(folder_path = user_path)
st.write('You selected `%s`' % filename)

df = load_data(filename)

data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = df
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')
st.subheader('Raw data')
st.write(data['comment'])
st.write('------------------------------')

# gameplay_sentence = "competitive interaction strategy tactics replay gameplay mechanic system elimination random decision tense deduction luck scoring dynamics"
# theme_sentence = 'art theme beautiful colorful design atmospheric universe'
# component_sentence = 'component layout marker tile box small large wear insert worn bits miniature minis production'
# length_sentence = 'slow quick fast filler setup takedown hours long'
# accessible_sentence = 'rule family learn easy hard difficult difficulty disability weight complex complicated accessible simple'


# dict_bg_tags_process = {'gameplay': gameplay_sentence.split(" "), 'theme': theme_sentence.split(" "), \
# 						'component': component_sentence.split(" "), 'length': length_sentence.split(" "), 'accessible': accessible_sentence.split(" ")}




st.header('2. Comment Sentiments')

FOLDER = './save/'
MODELFILE = 'model_tfid_rf_ncomments45000_nfeatures7500.sav'
VECTORFILE = 'vzer_ncomments45000_nfeatures7500.sav'

data = df.loc[df['comment'].str.count(' ') + 1 >= 50]
data['rating'] = df['rating'].astype(float)
data['rating'].loc[df['rating'] < 9.0] = 0
#data['rating'].loc[(df['rating'] > 4.0) & (df['rating'] < 9.0)] = np.nan
data['rating'].loc[df['rating'] >= 9.0] = 1
data = data.dropna()

# df['rating'].loc[(df['rating'] > 4.0) & (df['rating'] < 9.0)] = np.nan
# df['rating'].loc[df['rating'] < 9.0] = 0.0
# df['rating'].loc[df['rating'] >= 9.0] = 1.0
# df = df.dropna()

#print(get_sentiment(df['rating']))

pred = run_sentiment_model(data, FOLDER, MODELFILE, FOLDER, VECTORFILE)

num_positive = len(np.where(pred == 1.0)[0]) / len(pred) * 100
st.subheader(str(round(num_positive)) + '% of comments are predicted to rate your game highly on boardgamegeek.com (9 or 10 out of 10)')


st.write('_____________________________________________________________')


st.header('3. Categorization of Comments')
st.write('Three sample sentences are given that best highlight each category.')


data = df.loc[df['comment'].str.count(' ') + 1 >= 50]

category = 'Mechanics'
MODELCATFILE = 'model_tfid_category_log_hand_1_3_ncomments30000_nfeatures5000_' + category.lower() + '.sav' 
VECTORCATFILE = 'vzer_tfid_category_log_hand_1_3_ncomments30000_nfeatures5000_' + category.lower() + '.sav' 

pred, unique_features, unique_tdif, comments_likely = run_label_model(data, FOLDER, MODELCATFILE, FOLDER, VECTORCATFILE, top_n=750)

# data_category = data.loc[pred == 1]

# pred = run_sentiment_model(data_category, FOLDER, MODELFILE, FOLDER, VECTORFILE)

num_positive = len(np.where(pred == 1.0)[0]) / len(pred) * 100
st.subheader(str(round(num_positive)) + "% of comments mention the game's " + category.lower())
st.write(category + " refers to the gamplay and strategic elements of the game.")


count = 1
for comment in comments_likely:

	st.markdown('>' + comment)
	count += 1

st.write('_____________________________________________________________')

category = 'Theme'
MODELCATFILE = 'model_tfid_category_log_hand_1_3_ncomments30000_nfeatures5000_' + category.lower() + '.sav' 
VECTORCATFILE = 'vzer_tfid_category_log_hand_1_3_ncomments30000_nfeatures5000_' + category.lower() + '.sav' 

pred, unique_features, unique_tdif, comments_likely = run_label_model(data, FOLDER, MODELCATFILE, FOLDER, VECTORCATFILE, top_n=750)


#best_words = words[:5]
#best_tdif = tdif[:5]

num_positive = len(np.where(pred == 1.0)[0]) / len(pred) * 100
st.subheader(str(round(num_positive)) + "% of comments mention the game's " + category.lower() )
st.write(category + " refers to the game's theme and artistic design.")


count = 1
for comment in comments_likely:

	st.markdown('>' + comment)
	count += 1


#############
st.write('_____________________________________________________________')

category = 'Components'
MODELCATFILE = 'model_tfid_category_log_hand_1_3_ncomments30000_nfeatures5000_' + category.lower() + '.sav' 
VECTORCATFILE = 'vzer_tfid_category_log_hand_1_3_ncomments30000_nfeatures5000_' + category.lower() + '.sav' 
#category = 'Components'

pred, unique_features, unique_tdif, comments_likely = run_label_model(data, FOLDER, MODELCATFILE, FOLDER, VECTORCATFILE, top_n=750)
# data_category = data.loc[pred == 1]


#best_words = words[:5]
#best_tdif = tdif[:5]

num_positive = len(np.where(pred == 1.0)[0]) / len(pred) * 100
st.subheader(str(round(num_positive)) + "% of comments mention the game's " + category.lower() )
st.write(category + " refers to the game's physical quality, including durability and legibility.")


count = 1
for comment in comments_likely:

	st.markdown('>'  + comment)
	count += 1

st.write('_____________________________________________________________')

category = 'Length'
MODELCATFILE = 'model_tfid_category_log_hand_1_3_ncomments30000_nfeatures5000_' + category.lower() + '.sav' 
VECTORCATFILE = 'vzer_tfid_category_log_hand_1_3_ncomments30000_nfeatures5000_' + category.lower() + '.sav' 

pred, unique_features, unique_tdif, comments_likely = run_label_model(data, FOLDER, MODELCATFILE, FOLDER, VECTORCATFILE, top_n=750)
# data_category = data.loc[pred == 1]

# pred = run_sentiment_model(data_category, FOLDER, MODELFILE, FOLDER, VECTORFILE)
num_positive = len(np.where(pred == 1.0)[0]) / len(pred) * 100
st.subheader(str(round(num_positive)) + "% of comments mentioned the game's " + category.lower() )
st.write(category + " refers to the time spent playing the game.")

#best_words = words[:5]
#best_tdif = tdif[:5]

num_positive = len(np.where(pred == 1.0)[0]) / len(pred) * 100

count = 1
for comment in comments_likely:

	st.markdown('>' + comment)
	count += 1

st.write('_____________________________________________________________')

category = 'Accessibility'
MODELCATFILE = 'model_tfid_category_log_hand_1_3_ncomments30000_nfeatures5000_' + category.lower() + '.sav' 
VECTORCATFILE = 'vzer_tfid_category_log_hand_1_3_ncomments30000_nfeatures5000_' + category.lower() + '.sav' 

pred, unique_features, unique_tdif, comments_likely = run_label_model(data, FOLDER, MODELCATFILE, FOLDER, VECTORCATFILE, top_n=750)


#best_words = words[:5]
#best_tdif = tdif[:5]

num_positive = len(np.where(pred == 1.0)[0]) / len(pred) * 100
st.subheader(str(round(num_positive)) + "% of comments mentioned the game's " + category.lower() )
st.write(category + " refers to the game's complexity, if the rules are easy to learn, and if it's approachable to any demographic of gamer.")

count = 1
for comment in comments_likely:

	st.markdown('>' + comment)
	count += 1
