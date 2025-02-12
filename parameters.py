import torch

sourceFileName = 'drive/MyDrive/ML/en_bg_data/train.en'
targetFileName = 'drive/MyDrive/ML/en_bg_data/train.bg'
sourceDevFileName = 'drive/MyDrive/ML/en_bg_data/dev.en'
targetDevFileName = 'drive/MyDrive/ML/en_bg_data/dev.bg'
eng = "drive/MyDrive/ML/prep/Eng.txt"
bg = "drive/MyDrive/ML/prep/Bg.txt"

log_filename = "drive/MyDrive/ML/training_log8.csv"

bpe_Eng = "drive/MyDrive/ML/BPE/eng.model"
bpe_Bg = "drive/MyDrive/ML/BPE/bul.model"

corpusFileName = 'drive/MyDrive/ML/prep/corpusData2'
wordsFileName = 'drive/MyDrive/ML/prep/wordsData2'
modelFileName = 'drive/MyDrive/ML/prep/NMTmodel8'

device = torch.device("cuda:0")
# device = torch.device("cpu")

learning_rate = 0.001
batchSize = 16
clip_grad = 1.0

maxEpochs = 10
log_every = 10
test_every = 1000

emd_size = 128
hidden_size = 256
lstm_layers = 2

dropout_encoder = 0.15
dropout_translator = 0.15
dropaut_generator = 0.15
