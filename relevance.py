import typing
from pathlib import Path
import pandas as pd
import matchzoo as mz
import keras
import numpy as np



def load_data(path: str = 'train.csv') -> typing.Union[mz.DataPack, typing.Tuple[mz.DataPack, list]]:
	data_pack = mz.pack(pd.read_csv(path, index_col=0,error_bad_lines=False))

	data_pack.relation['label'] = data_pack.relation['label'].astype('float32')
	#print(len(data_pack))
	return data_pack




train_pack = load_data('train.csv')
valid_pack = load_data('dev.csv')
#predict_pack = load_data('test.csv')

'''
preprocessor = mz.preprocessors.DSSMPreprocessor()
train_processed = preprocessor.fit_transform(train_pack)
valid_processed = preprocessor.transform(valid_pack)


ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=4))
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]


model = mz.models.DSSM()
model.params['input_shapes'] = preprocessor.context['input_shapes']
model.params['task'] = ranking_task
model.params['mlp_num_layers'] = 3
model.params['mlp_num_units'] = 300
model.params['mlp_num_fan_out'] = 128
model.params['mlp_activation_func'] = 'relu'
model.guess_and_fill_missing_params()
model.build()
model.compile()


train_generator = mz.PairDataGenerator(train_processed, num_dup=1, num_neg=4, batch_size=64, shuffle=True)

valid_x, valid_y = valid_processed.unpack()
evaluate = mz.callbacks.EvaluateAllMetrics(model, x=valid_x, y=valid_y, batch_size=len(pred_x))

history = model.fit_generator(train_generator, epochs=20, callbacks=[evaluate], workers=5, use_multiprocessing=False)
'''

preprocessor = mz.preprocessors.BasicPreprocessor(remove_stop_words=True)
train_processed = preprocessor.fit_transform(train_pack)
test_processed = preprocessor.transform(valid_pack)

ranking_task = mz.tasks.Ranking(loss=mz.losses.RankHingeLoss())

ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]

#x = int(input('Enter a number1: '))

'''
model = mz.models.MVLSTM()
model.params.update(preprocessor.context)
model.params['task'] = ranking_task
model.params['embedding_output_dim'] = 300
model.params['lstm_units'] = 50
model.params['top_k'] = 20
model.params['mlp_num_layers'] = 2
model.params['mlp_num_units'] = 10
model.params['mlp_num_fan_out'] = 5
model.params['mlp_activation_func'] = 'relu'
model.params['dropout_rate'] = 0.5
model.params['optimizer'] = 'adadelta'
#model.guess_and_fill_missing_params()
model.build()
model.compile()
'''

'''
model = mz.models.DenseBaseline()
model.params.update(preprocessor.context)
model.params['task'] = mz.tasks.Ranking()
model.params['mlp_num_units'] = 10
model.build()
model.compile()
'''

model = mz.models.DSSM()
model.params.update(preprocessor.context)
model.params['task'] = mz.tasks.Ranking()
model.params['mlp_num_layers'] = 3
model.params['mlp_num_units'] = 300
model.params['mlp_num_fan_out'] = 128
model.params['mlp_activation_func'] = 'relu'
#model.guess_and_fill_missing_params()
model.build()
model.compile()


#x, y = train_processed.unpack()
pred_x, pred_y = test_processed.unpack()
#model.fit(x, y, batch_size=32, epochs=5)


evaluate = mz.callbacks.EvaluateAllMetrics(model, x=pred_x, y=pred_y, batch_size=len(pred_y))
print(len(pred_y))
x = int(input('Enter a number2: '))
data_generator = mz.DataGenerator(train_processed, batch_size=x)
print('num batches:', str(len(data_generator)))
x = int(input('Enter a number3: '))
history= model.fit_generator(data_generator, epochs=x, callbacks=[evaluate], use_multiprocessing=True, workers=4)

#x = int(input('Enter a number4: '))
model.save('my-model')

