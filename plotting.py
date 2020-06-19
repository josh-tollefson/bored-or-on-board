import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_year_published(year):

	# Plots numbers of games published every year

	fig, ax = plt.subplots(1, figsize=(8,6))

	ax.tick_params(which='major', direction='out', length=8, width=2, colors='k')
	ax.tick_params(axis='both', which='minor', direction='out', length=4, width=1, colors='k')
	ax.tick_params(axis='both', labelsize=20)
	ax.set_xlabel('Year', fontsize=22)
	ax.set_ylabel('Number of Board Games', fontsize=22)

	ax.hist(year.to_numpy(), bins=list(np.linspace(1950, 2021, 72)), edgecolor='black', linewidth=1.2, align='right')
	plt.savefig('/home/josh/Documents/Insights/Figures/bg-per-year.png', bbox_inches = "tight")
	plt.show()

def plot_owned(owned):

	fig, ax = plt.subplots(1, figsize=(8,6))

	ax.tick_params(which='major', direction='out', length=8, width=2, colors='k')
	ax.tick_params(axis='both', which='minor', direction='out', length=4, width=1, colors='k')
	ax.tick_params(axis='both', labelsize=20)
	ax.set_xlabel('Owned?', fontsize=22)
	ax.set_ylabel('Count', fontsize=22)

	ax.hist(owned.to_numpy(), bins=[0,1,2], edgecolor='black', linewidth=1.2, label=['Owned', 'Not Owned'])
	plt.show()


def plot_rated_vs_owned(owned, rating, binw=1):

	owned_rating = rating[owned == 1.0]
	not_owned_rating = rating[owned == 0.0]

	print(owned_rating)

	o = owned_rating.to_numpy(dtype="float16")
	uo = not_owned_rating.to_numpy(dtype="float16")

	bins = np.linspace(0,10,int((10-0)/binw + 1), endpoint=True)

	hist_o, bins_o = np.histogram(o[~np.isnan(o)], bins=bins, density=True) #weights=weights_o) ### For some reason, the weights did not give a sum of 1 under the curves
	hist_uo, bins_uo = np.histogram(uo[~np.isnan(uo)], bins=bins, density=True)

	fig, ax = plt.subplots(1, figsize=(8,6))

	ax.tick_params(which='major', direction='out', length=8, width=2, colors='k')
	ax.tick_params(axis='both', which='minor', direction='out', length=4, width=1, colors='k')
	ax.tick_params(axis='both', labelsize=20)
	ax.set_xlabel('Rating', fontsize=22)
	ax.set_ylabel('Fraction', fontsize=22)

	ax.plot(bins_o[1:], hist_o*binw, lw=3, marker='o', ms=10, label='Owned')
	ax.plot(bins_uo[1:], hist_uo*binw, lw=3, marker='o', ms=10, zorder=1, label='Not Owned')

	#ax.hist([owned_rating.to_numpy(), not_owned_rating.to_numpy()] , bins=[0,1,2,3,4,5,6,7,8,9,10], edgecolor='black', linewidth=1.2, stacked=False, \
	#	label=['Owned', 'Not Owned'], align='right', density=True)
	ax.legend(frameon=False, fontsize=16)
	plt.savefig('/home/josh/Documents/Insights/Figures/rating-vs-owned.png', bbox_inches = "tight")

	plt.show()

def plot_complexity_vs_owned(bg_owned, bg_unowned, binw=0.25):

	fig, ax = plt.subplots(1, figsize=(8,6))

	ax.tick_params(which='major', direction='out', length=8, width=2, colors='k')
	ax.tick_params(axis='both', which='minor', direction='out', length=4, width=1, colors='k')
	ax.tick_params(axis='both', labelsize=20)
	ax.set_xlabel('Complexity', fontsize=22)
	ax.set_ylabel('Fraciton', fontsize=22)

	o = bg_owned['complexity'].to_numpy(dtype="float16")
	uo = bg_unowned['complexity'].to_numpy(dtype="float16")

	bins = np.linspace(1,5,int((5-1)/binw + 1), endpoint=True)

	hist_o, bins_o = np.histogram(o[~np.isnan(o)], bins=bins, density=True)
	hist_uo, bins_uo = np.histogram(uo[~np.isnan(uo)], bins=bins, density=True)

	ax.plot(bins_o[1:], hist_o*binw, lw=3, marker='o', ms=10, label='Owned')
	ax.plot(bins_uo[1:], hist_uo*binw, lw=3, marker='o', ms=10, zorder=1, label='Not Owned')


	#ax.hist([co[~np.isnan(co)], cuo[~np.isnan(cuo)]] , bins=[1,1.5,2,2.5,3,3.5,4,4.5,5], edgecolor='black', linewidth=1.2, stacked=False, \
	#	label=['Owned', 'Not Owned'], align='right', weights=[weights_co, weights_cuo])
	ax.legend(frameon=False, fontsize=16)
	plt.savefig('/home/josh/Documents/Insights/Figures/complexity-vs-owned.png', bbox_inches = "tight")
	plt.show()

def plot_price_vs_owned(prices_owned, prices_unowned, binw=10):

	fig, ax = plt.subplots(1, figsize=(8,6))

	ax.tick_params(which='major', direction='out', length=8, width=2, colors='k')
	ax.tick_params(axis='both', which='minor', direction='out', length=4, width=1, colors='k')
	ax.tick_params(axis='both', labelsize=20)
	ax.set_xlabel('Price (USD)', fontsize=22)
	ax.set_ylabel('Fraction', fontsize=22)

	po = prices_owned.to_numpy(dtype="float16")
	puo = prices_unowned.to_numpy(dtype="float16")

	bins = np.linspace(0,160,(160-0)//binw + 1, endpoint=True)

	hist_po, bins_po = np.histogram(po[~np.isnan(po)], bins=bins, density=True)
	hist_puo, bins_puo = np.histogram(puo[~np.isnan(puo)], bins=bins, density=True)

	ax.plot(bins_po[1:], hist_po*binw, lw=3, marker='o', ms=10, label='Owned')
	ax.plot(bins_puo[1:], hist_puo*binw, lw=3, marker='o', ms=10, zorder=1, label='Not Owned')

	#ax.hist([po[~np.isnan(po)], puo[~np.isnan(puo)]] , bins=[0, 20, 40, 60, 80, 100, 120, 140, 160], edgecolor='black', linewidth=1.2, stacked=False, \
	#		 label=['Owned', 'Not Owned'], align='right', weights=[weights_po, weights_puo])
	ax.legend(frameon=False, fontsize=16)
	plt.savefig('/home/josh/Documents/Insights/Figures/price-vs-owned.png', bbox_inches = "tight")
	plt.show()


def plot_feature_and_runtime_vs_score(features, times, scores, log=False, baseline=[], feature_label='', savefile=''):

	fig, ax = plt.subplots(1, figsize=(8,8))

	#ax2 = ax.twinx()

	color1 = 'black'
	color2 = 'gray'

	if log:
		ax.set_xscale('log')

	ax.plot(features, scores, color=color1, lw=3, marker='o', markersize=10)
	#ax.plot(features, baseline, color='gray', lw=3, marker='o')
	#ax2.plot(features, times, color=color2, lw=2, marker='o')
	ax.set_xlabel(feature_label,fontsize=20)
	ax.set_ylabel("Accuracy",fontsize=20, color=color1)
	#ax2.set_ylabel("Runtime (s)",fontsize=20, color=color2)
	ax.tick_params(labelsize=18)
	ax.tick_params(axis='y',labelcolor=color1)
	#ax2.tick_params(axis='y',labelcolor=color2)
	#ax2.tick_params(labelsize=18, labelcolor=color2)
	plt.savefig(savefile, bbox_inches = "tight")
	plt.show()

def compare_models(times, scores, names, savefile=''):

	ds = pd.DataFrame({'scores': scores, 'times': times, 'names': names})
	fig, ax = plt.subplots(1, figsize=(8,8))

	g = sns.barplot( x='scores', y='names',data=ds, palette='Blues_r')
	g.set_yticklabels(ax.get_yticklabels(), rotation=0)
	g.set_ylabel("Model",fontsize=0)
	g.set_xlabel("Precision",fontsize=22)
	g.tick_params(labelsize=20)
	plt.savefig(savefile, bbox_inches = "tight")
	plt.show()

def plot_important_words(words, weights, savefile=''):

	ds = pd.DataFrame({'words': words, 'weights': weights})
	fig, ax = plt.subplots(1, figsize=(8,8))

	g = sns.barplot( x='weights', y='words',data=ds, palette='Blues_r')
	g.set_yticklabels(ax.get_yticklabels(), rotation=0)
	g.set_ylabel("Words",fontsize=20)
	g.set_xlabel("Weights",fontsize=22)
	g.tick_params(labelsize=20)
	plt.savefig(savefile, bbox_inches = "tight")
	plt.show()

## RF sentiment runs - ncomments

ncomments = [500, 1000, 5000, 10000, 20000, 30000, 45000]
times = [1.5, 2.6, 13.3, 35.4, 93.8, 200.7, 360.6]
accuracy = [0.72, 0.75, 0.81, 0.82, 0.85, 0.85, 0.86]
baseline = [0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52]

plot_feature_and_runtime_vs_score(ncomments, times, accuracy, feature_label='Number of Comments In Modeling', savefile='/home/josh/Documents/Insights/Figures/ncomments_time_vs_accuracy.png')

## MNB sentiment runs - nfeatures in vectorizer

nfeatures = [50, 100, 500, 1000, 5000, 7500, 20000, 50000]
times = [0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 1.0]
accuracy = [0.67, 0.72, 0.8, 0.84, 0.86, 0.86, 0.87, 0.86]
baseline = [0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52]

plot_feature_and_runtime_vs_score(nfeatures, times, accuracy, log=True, feature_label='Number of Word Features In Modeling', savefile='/home/josh/Documents/Insights/Figures/nfeatures_time_vs_accuracy.png')


## MNB sentiment runs - nfeatures in vectorizer

nfeatures = [1, 10, 25, 50, 100, 500]
times = [0.8, 0.9, 0.9, 0.9, 0.9, 0.9]
accuracy = [0.81, 0.84, 0.85, 0.87, 0.88, 0.81]
baseline = [0.52, 0.52, 0.52, 0.52, 0.52, 0.52]

plot_feature_and_runtime_vs_score(nfeatures, times, accuracy, log=True, feature_label='Minimum Number of Words In Comment', savefile='/home/josh/Documents/Insights/Figures/nwords_accuracy.png')


## Baseline vs RF vs MNB sentiment runs - nfeatures in vectorizer
## Baseline, MNB, RF

times = [0, 33, 360, 76]
accuracy = [0.52, 0.87, 0.86, 0.88]
names = ['Baseline', 'Multi. Naive Bayes', 'Random Forest', 'Logistic']

compare_models(times, accuracy, names, savefile='/home/josh/Documents/Insights/Figures/model_comparison_runtime.png')

## Compare model precisions

times = [0, 0, 0, 0]
precision = [0.44, 0.70, 0.73, 0.25]
error = [0.017, 0.014, 0.019, 0.0]
names = ['Multi. Naive Bayes', 'Logistic', 'Random Forest','Baseline']

compare_models(times, precision, names, savefile='/home/josh/Documents/Insights/Figures/model_comparison_precision_copmonents.png')

### Compare important words

words_all = ['monopoly', 'trade', 'rule', 'people', 'house', 'player']
weights_all = [0.02305592, 0.00881587, 0.00838563, 0.0073771,  0.00700961, 0.00657115]
words_unique = ['trade', 'roll', 'luck', 'land', 'auction', 'move']
weights_unique = [0.008815865896135982, 0.006354208216731602, 0.005699365012257282, 0.005553120280156337, 0.004909252511716566, 0.0045987856685923345]

words_general = ['monopoly', 'was', 'family', 'year', 'time', 'board']
weights_general = [0.03013665, 0.00937369, 0.00823223, 0.0081202,  0.00750712, 0.00715268]

plot_important_words(words_unique, weights_unique, savefile='/home/josh/Documents/Insights/Figures/monopoly_mechanics_words_1_unique.png')