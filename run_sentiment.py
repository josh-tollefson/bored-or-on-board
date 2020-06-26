import re
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import utils
from wordprocess import preprocessor
import timeit

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import svm
from xgboost import XGBClassifier, XGBRFClassifier
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression


def get_sentiment(ratings, min_rating = 4.0, max_rating = 9.0):
	'''
	Filter dataframe based on extreme ratings to set up sentiment analysis
	based on 0 or 1 labels for future classification work
	INPUT:
		ratings: dataframe, column of user ratings
		min_rating: float, minimum rating for filtering
		max_rating: float, maximum rating for filtering
	OUTPUT:
		sentiment, dataframe column with the filtered 0 or 1 sentiment
	'''

	df_sentiment = pd.DataFrame(0, index=np.arange(len(ratings.to_numpy())), columns=['sentiment'])

	df_sentiment['sentiment'].loc[ratings <= min_rating] = 0
	df_sentiment['sentiment'].loc[(ratings > min_rating) & (ratings < max_rating)] = np.nan
	df_sentiment['sentiment'].loc[ratings >= max_rating] = 1

	return df_sentiment['sentiment']



def freq_words(allcomments):

	all_words = ' '.join([text for text in allcomments])
	all_words = all_words.split()
	fdist = nltk.FreqDist(all_words)
	words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

	d = words_df.nlargest(columns="count", n = 20)


	plt.figure(figsize=(12,15))
	ax = sns.barplot(data=d, x="count", y="word")
	ax.set(ylabel = 'word')
	plt.show()

def get_all_comments(infile):

	try:
		df = pd.read_csv(infile)

		allcomments = []
		for comment in df['comment']:
			allcomments.append(comment)

		return allcomments

	except pd.errors.ParserError:
		print('Need CSV file')

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=50):
    """return n-gram counts in descending order of counts"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    results=[]
    
    # word index, count i
    for idx, count in sorted_items:
        
        # get the ngram name
        n_gram=feature_names[idx]
        
        # collect as a list of tuples
        results.append((n_gram,count))
 
    return results



def run_binary_class_model(comments_df):

	from sklearn.model_selection import train_test_split
	from keras.models import Sequential
	from keras.layers import Dense
	from keras.layers import Dropout

	X_train, X_test, Y_train, Y_test = train_test_split(comments_df.iloc[:,4:], comments_df.iloc[:,3], test_size=0.20, random_state=42)

	n_words = X_test.shape[1]

	#run_pca(X_test, X_train)

	# define network
	model = Sequential()
	model.add(Dense(300, input_shape=(n_words,), activation='relu'))
	model.add(Dense(100, input_shape=(n_words,), activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# compile network
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()
	model.fit(X_train, Y_train, epochs=20, validation_data=(X_test, Y_test), verbose=2)
	# evaluate
	loss, acc = model.evaluate(X_test, Y_test, verbose=0)
	#scores.append(acc)
	print('accuracy: %s' % acc)
	predictions = model.predict(X_test)

	print ("\nclassification report :\n",(confusion_matrix(Y_test,np.round(predictions))))

	from sklearn.svm import SVC
	SVM_model=SVC()
	SVM_model.fit(X_train,Y_train)


	#scores.append(acc)
	predictions = SVM_model.predict(X_test)
	print(predictions)

	print ("\nclassification report :\n",(confusion_matrix(Y_test,np.round(predictions))))


	return model

def run_pca(X_test, X_train):

	from sklearn.decomposition import PCA
	pca = PCA() 
	X_train = pca.fit_transform(X_train)
	X_test = pca.transform(X_test)
	total=sum(pca.explained_variance_)
	k=0
	current_variance=0
	while current_variance/total < 0.99:
		current_variance += pca.explained_variance_[k]
		k=k+1

	print(k)

	pca = PCA(n_components=k)
	X_train = pca.fit_transform(X_train)
	X_test = pca.transform(X_test)
	plt.figure()
	plt.plot(np.cumsum(pca.explained_variance_ratio_))
	plt.xlabel('Number of Components')
	plt.ylabel('Variance (%)') #for each component
	plt.title('Exoplanet Dataset Explained Variance')
	plt.show()

def read_jl_files(inpath, nfiles=3, comment_length=25):

	import glob
	count = 0
	df_all = pd.DataFrame(columns=['comment', 'rating', 'bgg_id'])

	for file in glob.glob(inpath + '*'):

		if count < nfiles:

			df = pd.read_json(file, lines=True)
			df_processed = process_df(df, comment_length=comment_length)
			df_all = df_all.append(df_processed)

			count += 1

		else:

			break

		print('READING FILE %d / %d' % (count,nfiles))

	df_all = df_all.sample(frac=1).reset_index(drop=True)

	return df_all


def process_df(df, comment_length=25):

	# Only want to work with some of the columns
	# We have lots of data, so work with only some of it to save memory.
	
	allcomments = df[df['comment'].notna()]['comment'] 
	allratings = df[df['comment'].notna()]['bgg_user_rating']
	allbggids = df[df['comment'].notna()]['bgg_id']

	allcomments_re = allcomments.loc[allcomments.str.count(' ') + 1 >= comment_length]
	allratings_re = allratings.loc[allcomments.str.count(' ') + 1  >= comment_length]
	allbggids_re = allbggids.loc[allcomments.str.count(' ') + 1  >= comment_length]
	alluserids_re = allbggids.loc[allcomments.str.count(' ') + 1  >= comment_length]


	return pd.DataFrame({'comment': allcomments_re.values, 'rating': allratings_re.values, 'bgg_id': allbggids_re.values})

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(X, features, row_id, top_n=50):
	''' Top tfidf features in specific document (matrix row) '''
	row = np.squeeze(X[row_id].toarray())
	return top_tfidf_feats(row, features, top_n)

def top_mean_feats(X, features, grp_ids=None, min_tfidf=0.001, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = X[grp_ids].toarray()
    else:
        D = X.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def plot_labels(words, tdif):

	words_df = pd.DataFrame({"words": words, "tdif": tdif})
	sns.barplot(x='words', y='tdif', data=words_df, orient="h")
	del words_df
	plt.show()

def plot_comment_length(df):

	df['comment_length'] = df.loc[df['comment'].notna()]['comment'].str.count(' ') + 1

	fig, ax = plt.subplots(1, figsize=(8,6))

	ax.tick_params(which='major', direction='out', length=8, width=2, colors='k')
	ax.tick_params(axis='both', which='minor', direction='out', length=4, width=1, colors='k')
	ax.tick_params(axis='both', labelsize=20)

	logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins)); 
	ax.hist(df['comment_length'].values, bins=logbins)
	ax.set_xscale('log')
	ax.set_xlabel('# Words', fontsize=22)
	ax.set_ylabel('# Comments', fontsize=22)
	plt.savefig('/home/josh/Documents/Insights/Figures/comment_count_hist.png', bbox_inches = "tight")
	plt.show()


def plot_word_frequency(vectorizer, Xwords, savefile='/home/josh/Documents/Insights/Figures/word_freq_hist.png'):

	sum_words = Xwords.sum(axis=0)
	words_freq = [[word, sum_words[0, idx]] for word, idx in vectorizer.vocabulary_.items()]
	words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

	common_words = [item[0] for item in words_freq] 
	common_sums = [item[1] for item in words_freq] 

	ds_word_freq = pd.DataFrame({'words': common_words[:20], 'number': common_sums[:20]})

	fig, ax = plt.subplots(1, figsize=(8,8))

	print(words_freq[:20])
	g = sns.barplot( x='number', y='words',data=ds_word_freq, orient="h", palette='Blues_d')
	g.set_yticklabels(ax.get_yticklabels(), rotation=0)
	g.set_ylabel("Words",fontsize=20)
	g.set_xlabel("Count",fontsize=20)
	g.tick_params(labelsize=16)
	plt.savefig(savefile, bbox_inches = "tight")
	plt.show()


def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}
    
    # loop for each class
    classes ={}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i,el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key = lambda x : x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key = lambda x : x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops':tops,
            'bottom':bottom
        }
    return classes

def plot_important_words(top_scores, top_words, bottom_scores, bottom_words, name):
    y_pos = np.arange(len(top_words))
    top_pairs = [(a,b) for a,b in zip(top_words, top_scores)]
    top_pairs = sorted(top_pairs, key=lambda x: x[1])
    
    bottom_pairs = [(a,b) for a,b in zip(bottom_words, bottom_scores)]
    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)
    
    top_words = [a[0] for a in top_pairs]
    top_scores = [a[1] for a in top_pairs]
    
    bottom_words = [a[0] for a in bottom_pairs]
    bottom_scores = [a[1] for a in bottom_pairs]
    
    fig = plt.figure(figsize=(10, 10))  

    plt.subplot(121)
    plt.barh(y_pos,bottom_scores, align='center', alpha=0.5)
    plt.title('Irrelevant', fontsize=20)
    plt.yticks(y_pos, bottom_words, fontsize=14)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    
    plt.subplot(122)
    plt.barh(y_pos,top_scores, align='center', alpha=0.5)
    plt.title('Disaster', fontsize=20)
    plt.yticks(y_pos, top_words, fontsize=14)
    plt.suptitle(name, fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    
    plt.subplots_adjust(wspace=0.8)
    plt.show()


def run_model(model, X, Y):
	'''
	Train and return model parameters for sentiment analysis
	INPUTS: 
			model, sklearn model
			X, array of vectorized comments
			Y, array of sentiment labels (0, 1)
	OUTPUTS:
			best_grid: optimum best_grid model based on hyperparam. tuning
			score: float, accuracy of best model
			report: classification report of best model
			runtime: float, total runtime (s) to train model
			baseline_acc: int of baseline accuracy based on random sampling
	'''

	zero_label = list(Y).count(0)
	one_label = list(Y).count(1)
	baseline_acc = (zero_label / (zero_label + one_label))**2 + (one_label / (zero_label + one_label))**2

	start = timeit.default_timer()

	from sklearn.model_selection import train_test_split
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

	grid_search = run_grid_search('rf', model)
	grid_search.fit(X_train, Y_train)
	best_grid = grid_search.best_estimator_
	accuracy = best_grid.score(X_test, Y_test)
	params = grid_search.best_params_
	score = grid_search.best_score_

	stop = timeit.default_timer()

	runtime = stop - start


	Y_pred = model.predict(X_test)

	report = classification_report(Y_test, Y_pred, output_dict=True)
	return best_grid, score, report, runtime, baseline_acc


def run_grid_search(modeltype, model):
	'''
	Return grid_search based on model parameters for random forest only (currently)
	INPUT:
		modeltype: str, type of model ('rf' for randomforest currently)
		model: initialied sklearn model
	OUTPUT:
		grid of model and hyperpararmeters
	'''
	from sklearn.model_selection import GridSearchCV

	if modeltype == 'rf':
		param_grid = {
			'bootstrap': [True],
			'max_depth': [10, 50, 100],
			'max_features': [2, 10, 'sqrt'],
			'min_samples_leaf': [1, 3, 5],
			'min_samples_split': [2, 6, 10],
			'n_estimators': [10, 100, 1000]
		}

	grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

	return grid_search

# comments_length=0
# inpath = '/home/josh/Documents/Code/Projects/board-game-filler/files/'
# df = read_jl_files(inpath, nfiles=10, comment_length=comments_length)
# sentiments = get_sentiment(df['rating'], min_rating=4.0, max_rating=9.0)
# df['sentiment'] = sentiments
# df = df.dropna()#.sort_values(by='bgg_id')

NCOMMENTS = 10000
nmax = 45000
max_features = 7500

FOLDER = './save/'
INPUT_PANDAS = 'comments_extreme.jl'
OUTPUT_MODEL = 'model_sentiment_tfid_logreg_ncomments' + str(NCOMMENTS) + '_nfeatures' + str(max_features) + '.sav' 
INPUT_VZER = 'vzer_ncomments' + str(nmax) + '_nfeatures' + str(max_features) + '.sav' 
#OUTPUT_PANDAS = 'comments_extreme_nlength25.jl'
OUTPUT_REPORT = 'report_sentiment_tfid_logreg_ncomments' + str(NCOMMENTS) + '_nfeatures' + str(max_features) + '.csv'


df = utils.load_dataframe(FOLDER, INPUT_PANDAS)
df = df.dropna()

#model = RandomForestClassifier(n_estimators=200, 
#                               bootstrap = True,
#                               max_features = 'sqrt')
model = RandomForestClassifier()
#model = MultinomialNB()

comments_reduced = df.iloc[:NCOMMENTS]

#model = LogisticRegression()

#Tfmer = TfidfVectorizer(sublinear_tf=True, max_features=max_features, ngram_range=(1, 4), preprocessor=preprocessor)
Tfmer = utils.load_model(FOLDER, INPUT_VZER)

#############################

X_transformed = Tfmer.fit_transform(comments_reduced['comment'].values)
Y = comments_reduced['sentiment'].values

model, score, report, runtime, baseline_stats = run_model(model, X_transformed, Y, kfolds=None)

#utils.save_model(models[-1], FOLDER, OUTPUT_MODEL)
#utils.save_model(Tfmer, FOLDER, OUTPUT_MODEL)
#utils.save_report(report, FOLDER, OUTPUT_REPORT, runtime=runtime, baseline=baseline_stats, stddev=stddev)
#utils.save_dataframe('./save/comments_regression.jl')

