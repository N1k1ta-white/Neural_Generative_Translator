#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2024/2025
##########################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import sys
import random
import nltk
from nltk.translate.bleu_score import corpus_bleu

import sentencepiece as spm
import os

nltk.download('punkt')

def train_bpe(corpus_file, model_prefix, vocab_size=8000):
    if not os.path.exists(f"{model_prefix}.model"):
        spm.SentencePieceTrainer.train(
            input=corpus_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type="bpe",
            bos_id=0,
            eos_id=1,
            unk_id=2,
            pad_id=3,
        )


def get_word2ind(sp_model):
    trainCorpusBg, trainCorpusEng, devCorpusBg, devCorpusEng, sp_source, sp_target = prepareDataBPE(sourceFileName, targetFileName, sourceDevFileName, targetDevFileName, bpe_Eng, bpe_Bg)
    word2indBg = get_word2ind(sp_target)
    word2indEng = get_word2ind(sp_source)

    pickle.dump((trainCorpusBg, trainCorpusEng, devCorpusBg, devCorpusEng), open(corpusFileName, 'wb'))
    pickle.dump((word2indEng, word2indBg), open(wordsFileName, 'wb'))

    print('Data prepared.')

# train_bpe(bg, "bul", 40000)
# train_bpe(eng, "eng", 30000)

class progressBar:
    def __init__(self ,barWidth = 50):
        self.barWidth = barWidth
        self.period = None
    def start(self, count):
        self.item=0
        self.period = int(count / self.barWidth)
        sys.stdout.write("["+(" " * self.barWidth)+"]")
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.barWidth+1))
    def tick(self):
        if self.item>0 and self.item % self.period == 0:
            sys.stdout.write("-")
            sys.stdout.flush()
        self.item += 1
    def stop(self):
        sys.stdout.write("]\n")

def readCorpus(fileName):
    ### Чете файл от изречения разделени с нов ред `\n`.
    ### fileName е името на файла, съдържащ корпуса
    ### връща списък от изречения, като всяко изречение е списък от думи
    print('Loading file:',fileName)
    return [ nltk.word_tokenize(line) for line in open(fileName, 'r', encoding='utf-8') ]

def getDictionary(corpus, startToken, endToken, unkToken, padToken, transToken, wordCountThreshold = 2):
    dictionary={}
    for s in corpus:
        for w in s:
            if w in dictionary: dictionary[w] += 1
            else: dictionary[w]=1

    words = [startToken, endToken, unkToken, padToken, transToken] + [w for w in sorted(dictionary) if dictionary[w] > wordCountThreshold]
    return { w:i for i,w in enumerate(words)}

def MyPrepareData(sourceFileName, targetFileName, sourceDevFileName, targetDevFileName, startToken, endToken, unkToken, padToken, transToken):
    sourceCorpus = readCorpus(sourceFileName)
    targetCorpus = readCorpus(targetFileName)
    word2indEng = getDictionary(sourceCorpus, startToken, endToken, unkToken, padToken, transToken)
    word2indBg = getDictionary(targetCorpus, startToken, endToken, unkToken, padToken, transToken)

    trainCorpusBg = [ [startToken] + s + [endToken] for s in targetCorpus]
    trainCorpusEn = [ [startToken] + s + [endToken] for s in sourceCorpus]

    sourceDev = readCorpus(sourceDevFileName)
    targetDev = readCorpus(targetDevFileName)

    devCorpusBg = [ [startToken] + s + [endToken] for s in targetDev]
    devCorpusEn = [ [startToken] + s + [endToken] for s in sourceDev]

    print('Corpus loading completed.')
    return trainCorpusBg, trainCorpusEn, devCorpusBg, devCorpusEn, word2indEng, word2indBg

def prepareData(sourceFileName, targetFileName, sourceDevFileName, targetDevFileName, startToken, endToken, unkToken, padToken, transToken):

    sourceCorpus = readCorpus(sourceFileName)
    targetCorpus = readCorpus(targetFileName)
    word2ind = getDictionary(sourceCorpus+targetCorpus, startToken, endToken, unkToken, padToken, transToken)

    trainCorpus = [ [startToken] + s + [transToken] + t + [endToken] for (s,t) in zip(sourceCorpus,targetCorpus)]

    sourceDev = readCorpus(sourceDevFileName)
    targetDev = readCorpus(targetDevFileName)

    devCorpus = [ [startToken] + s + [transToken] + t + [endToken] for (s,t) in zip(sourceDev,targetDev)]

    print('Corpus loading completed.')
    return trainCorpus, devCorpus, word2ind

def load_bpe(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp

def encode_bpe(sp, sentences):
    return [sp.encode(s, out_type=int) for s in sentences]

def listComprehension(nested_list):
  flattened_list = [item for sublist in nested_list for item in sublist]
  return flattened_list

def prepareDataBPE(sourceFileName, targetFileName, sourceDevFileName, targetDevFileName, bpe_model_source, bpe_model_target, startTokenIdx=0, endTokenIdx=1):
    sourceCorpus = readCorpus(sourceFileName)
    targetCorpus = readCorpus(targetFileName)

    sp_source = load_bpe(bpe_model_source)
    sp_target = load_bpe(bpe_model_target)

    trainCorpusEng = [[startTokenIdx] + listComprehension(s) + [endTokenIdx] for s in encode_bpe(sp_source, sourceCorpus)]
    trainCorpusBg = [[startTokenIdx] + listComprehension(s) + [endTokenIdx] for s in encode_bpe(sp_target, targetCorpus)]

    sourceDev = readCorpus(sourceDevFileName)
    targetDev = readCorpus(targetDevFileName)

    devCorpusEng = [[startTokenIdx] + listComprehension(s) + [endTokenIdx] for s in encode_bpe(sp_source, sourceDev)]
    devCorpusBg = [[startTokenIdx] + listComprehension(s) + [endTokenIdx] for s in encode_bpe(sp_target, targetDev)]

    print('Corpus loading and BPE encoding completed.')
    return trainCorpusBg, trainCorpusEng, devCorpusBg, devCorpusEng, sp_source, sp_target
