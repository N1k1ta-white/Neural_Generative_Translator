import torch

sourceFileName = 'drive/MyDrive/en_bg_data/train.en'
targetFileName = 'drive/MyDrive/en_bg_data/train.bg'
sourceDevFileName = 'drive/MyDrive/en_bg_data/dev.en'
targetDevFileName = 'drive/MyDrive/en_bg_data/dev.bg'
eng = "drive/MyDrive/prep/Eng.txt"
bg = "drive/MyDrive/prep/Bg.txt"

bpe_Eng = "drive/MyDrive/BPE/eng.model"
bpe_Bg = "drive/MyDrive/BPE/bul.model"

corpusFileName = 'drive/MyDrive/prep/corpusData'
wordsFileName = 'drive/MyDrive/prep/wordsData'
modelFileName = 'drive/MyDrive/prep/NMTmodel'

device = torch.device("cuda:0")
#device = torch.device("cpu")

learning_rate = 0.001
batchSize = 16
clip_grad = 1.0

maxEpochs = 10
log_every = 10
test_every = 1000

emd_size = 128
hidden_size = 256
lstm_layers = 2

dropout_encoder = 0.2
dropout_translator = 0.15
dropaut_generator = 0.15
dropout_attention = 0.2

scheduled_sampling_rate = 0.3