# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def get_gru_result():
	gru_results = {}
	for idx in range(2, 7):
		acc = []
		val_acc = []
		loss = []
		val_loss = []

		with open('../results/result-gru-with_seq_len-%d.txt' % idx) as fd:
			lines = fd.readlines()
			acc = list( map(float, lines[0].split()) )
			val_acc = list( map(float, lines[1].split()) )
			loss = list( map(float, lines[2].split()) )
			val_loss = list( map(float, lines[3].split()) )

		gru_results[idx] = { 'acc': acc, 'val_acc': val_acc, 'loss': loss, 'val_loss': val_loss }

	return gru_results

	
def plot_gru_result():
	gru_results = get_gru_result()

	# plot train acc
	fig = plt.figure()
	plt.xlabel('epoch')
	plt.ylabel('train accuracy')	
	for idx in range(2, 7):
		plt.plot( gru_results[idx]['acc'], label=r'seq_len=%d'%idx )
	plt.legend(loc='best')
	plt.savefig('../plots/gru_train_acc_results.png')
	plt.close('all')

	# plot validate acc
	fig = plt.figure()
	plt.xlabel('epoch')
	plt.ylabel('validate accuracy')	
	for idx in range(2, 7):
		plt.plot( gru_results[idx]['val_acc'], label=r'seq_len=%d'%idx )
	plt.legend(loc='best')
	plt.savefig('../plots/gru_val_acc_results.png')
	plt.close('all')

	# plot train loss
	fig = plt.figure()
	plt.xlabel('epoch')
	plt.ylabel('train loss')	
	for idx in range(2, 7):
		plt.plot( gru_results[idx]['loss'], label=r'seq_len=%d'%idx )
	plt.legend(loc='best')
	plt.savefig('../plots/gru_train_loss_results.png')
	plt.close('all')

	# plot validate loss
	fig = plt.figure()
	plt.xlabel('epoch')
	plt.ylabel('validate loss')	
	for idx in range(2, 7):
		plt.plot( gru_results[idx]['val_loss'], label=r'seq_len=%d'%idx )
	plt.legend(loc='best')
	plt.savefig('../plots/gru_val_loss_results.png')
	plt.close('all')


def get_lstm_result():
	lstm_results = {}
	for idx in range(2, 7):
		acc = []
		val_acc = []
		loss = []
		val_loss = []

		with open('../results/result-lstm-with_seq_len-%d.txt' % idx) as fd:
			lines = fd.readlines()
			acc = list( map(float, lines[0].split()) )
			val_acc = list( map(float, lines[1].split()) )
			loss = list( map(float, lines[2].split()) )
			val_loss = list( map(float, lines[3].split()) )

		lstm_results[idx] = { 'acc': acc, 'val_acc': val_acc, 'loss': loss, 'val_loss': val_loss }

	return lstm_results

	
def plot_lstm_result():
	lstm_results = get_lstm_result()

	# plot train acc
	fig = plt.figure()
	plt.xlabel('epoch')
	plt.ylabel('train accuracy')	
	for idx in range(2, 7):
		plt.plot( lstm_results[idx]['acc'], label=r'seq_len=%d'%idx )
	plt.legend(loc='best')
	plt.savefig('../plots/lstm_train_acc_results.png')
	plt.close('all')

	# plot validate acc
	fig = plt.figure()
	plt.xlabel('epoch')
	plt.ylabel('validate accuracy')	
	for idx in range(2, 7):
		plt.plot( lstm_results[idx]['val_acc'], label=r'seq_len=%d'%idx )
	plt.legend(loc='best')
	plt.savefig('../plots/lstm_val_acc_results.png')
	plt.close('all')

	# plot train loss
	fig = plt.figure()
	plt.xlabel('epoch')
	plt.ylabel('train loss')	
	for idx in range(2, 7):
		plt.plot( lstm_results[idx]['loss'], label=r'seq_len=%d'%idx )
	plt.legend(loc='best')
	plt.savefig('../plots/lstm_train_loss_results.png')
	plt.close('all')

	# plot validate loss
	fig = plt.figure()
	plt.xlabel('epoch')
	plt.ylabel('validate loss')	
	for idx in range(2, 7):
		plt.plot( lstm_results[idx]['val_loss'], label=r'seq_len=%d'%idx )
	plt.legend(loc='best')
	plt.savefig('../plots/lstm_val_loss_results.png')
	plt.close('all')


def plot_gru_lstm_val_acc():
	gru_result = get_gru_result()
	lstm_results = get_lstm_result()

	for idx in range(2, 7):
		fig = plt.figure()
		plt.xlabel('epoch')
		plt.ylabel('validate accuracy')	
		plt.title('GRU and LSTM model accuracy with seq_len = %d' % idx)
		plt.plot(gru_result[idx]['val_acc'], label='GRU', marker='d')
		plt.plot(lstm_results[idx]['val_acc'], label='LSTM', marker='^', linestyle=':')
		plt.legend(loc='best')
		plt.savefig('gru_and_lstm_model_val_acc-%d.png' % idx)
		plt.close('all')

	fig = plt.figure(figsize=(15,10), dpi=80)
	plt.title('GRU and LSTM model accuracy with different sequeen length')
	for idx in range(2, 6):
		plt.subplot(2, 2, idx-1)
		plt.title('sequence length = %d' % idx)
		plt.xlabel('epoch')
		plt.ylabel('accuracy')
		plt.plot(gru_result[idx]['val_acc'], label='GRU', marker='d')
		plt.plot(lstm_results[idx]['val_acc'], label='LSTM', marker='^', linestyle=':')
		plt.legend(loc='upper left')
	plt.savefig('gru_and_lstm_model_val_acc.png' )
	plt.close('all')

	fig = plt.figure(figsize=(15,10), dpi=80)
	for idx in range(2, 6):
		plt.subplot(2, 2, idx-1)
		plt.title('sequence length = %d' % idx)
		plt.xlabel('epoch')
		plt.ylabel('loss')
		plt.plot(gru_result[idx]['val_loss'], label='GRU', marker='d')
		plt.plot(lstm_results[idx]['val_loss'], label='LSTM', marker='^', linestyle=':')
		plt.legend(loc='upper left c                                                                                   5555555555555555555555555555552')
	plt.savefig('gru_and_lstm_model_val_loss.png' )
	plt.close('all')

if __name__ == '__main__':
	#plot_gru_result()
	#plot_lstm_result()
	plot_gru_lstm_val_acc()