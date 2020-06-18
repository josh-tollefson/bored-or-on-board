import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import itertools

### Generally useful functions for plotting and saving models ####


def get_bgg_from_bga(bga_game, bgg_data):

	# Return the relevant BGG data row from a BGA (board game atlas) name

	bgg_game = bgg_data[bgg_data['name'] == bga_game]

	print(bgg_game)

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


def get_bgg_from_user(user_bgg_ids, bgg_data):

	bgg_data.sort_value(by='bgg_id')

	bg = bgg_data[bgg_data['bgg_id'].isin(user_bgg_ids)]

	return bg


def get_price_from_user(user_bgg_ids, bga_data):

	bg = bgg_data[bga_data['bgg_id'].isin(user_bgg_ids)]

	return bg

def separate_owned_bgs(user_data, bgg_data):

	user_bgg_owned_ids = user_data[user_data['bgg_user_owned'] == 1.0]['bgg_id'].values
	user_bgg_unowned_ids = user_data[user_data['bgg_user_owned'] == 0.0]['bgg_id'].values

	bg_owned = get_bgg_from_user(user_bgg_owned_ids, bgg_data)
	bg_unowned = get_bgg_from_user(user_bgg_unowned_ids, bgg_data)

	return bg_owned, bg_unowned

def strip_currency(prices):

	prices = prices.str.split(pat = "D", expand=True)[1]
	pd.to_numeric(prices)

	return prices

def load_dataframe(folder, infile):

	if infile.split('.')[-1] == 'jl':

		df = pd.read_json(folder + infile).reset_index(drop=True)

	elif infile.split('.')[-1] == 'csv':

		df = pd.read_csv(folder + infile).reset_index(drop=True)

	return df


def save_dataframe(df, folder, outfile):

	if outfile.split('.')[-1] == 'jl':

		df.to_json(folder + outfile)

	elif outfile.split('.')[-1] == 'csv':

		df.to_csv(folder + outfile)

	return df


def load_model(folder, infile):
	''' Used for both sklearn models
		and tfidf vectorizers.'''

	model = pickle.load( open(folder + infile, 'rb') )

	return model

def save_model(model, folder, outfile):
	''' Used for both sklearn models
		and tfidf vectorizers.'''

	pickle.dump(model, open(folder + outfile, 'wb'))

def save_report(report, folder, outfile, runtime='', baseline='', stddev=''):

	report_df = pd.DataFrame(report).transpose()
	report_df['runtime'] = runtime
	report_df['baseline'] = baseline
	report_df['stddev'] = stddev
	report_df.to_csv(folder + outfile)


# bga_file = "/home/josh/Documents/Code/Projects/bgg/board-game-scraper/feeds/bga/GameItem/2020-05-28T16-27-50.jl"
# bgg_file = "/home/josh/Documents/Code/Projects/bgg/board-game-scraper/feeds/bgg/GameItem/2020-05-27T05-38-16.jl"
# #user_file = '/home/josh/Documents/Code/Projects/board-game-filler/test.jl'

# # #bga_data = pd.read_json(bga_file, lines=True)
# bgg_data = pd.read_json(bgg_file, lines=True)
# #user_data = pd.read_json(user_file, lines=True)
# #print(bgg_data['mechanic'].head(10))

# ids = bgg_data.bgg_id.unique()

# d = {i:0 for i in ids}

# print(d)


#plt.scatter(bgg_data['avg_rating'].values, bgg_data['rating'].values)

#bga_names = bga_data['name']
#bga_price = bga_data['list_price']

#print(bgg_data.info())

#print(bgg_data[bgg_data['year'] == 2021])


#bg_owned, bg_unowned = separate_owned_bgs(user_data, bgg_data)

#price_owned = bga_price[bga_names.isin(bg_owned['name'])]
#price_unowned = bga_price[bga_names.isin(bg_unowned['name'])]
#prices_owned = strip_currency(price_owned)
#prices_unowned = strip_currency(price_unowned)

#print(prices_owned)

#plot_price_vs_owned(prices_owned, prices_unowned)

#plot_complexity_vs_owned(bg_owned, bg_unowned)


#owned = user_data['bgg_user_owned']
#rating = user_data['bgg_user_rating']

#year = bgg_data['year']

#plot_year_published(year)

#plot_rated_vs_owned(owned, rating)
#plot_owned(owned)

#bgg_data.info()

#bga_game = bga_data['name'][0]


#get_bgg_from_bga(bga_data, bgg_data)

