#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2024/2025
#############################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import sys
import numpy as np
import torch
import math
import pickle
import time

from nltk.translate.bleu_score import corpus_bleu

import utils
import model
from parameters import *

startToken = '<s>'
startTokenIdx = 0

endToken = '</s>'
endTokenIdx = 1

unkToken = '<unk>'
unkTokenIdx = 2

padToken = '<pad>'
padTokenIdx = 3

transToken = '<trans>'
transTokenIdx = 4

def perplexity(nmt, testEng, testBg, batchSize):
    testSize = min(len(testEng), len(testBg))
    total_loss = 0.0
    total_words = 0
    
    nmt.eval()  # Set model to evaluation mode
    
    for b in range(0, testSize, batchSize):
        eng_batch = testEng[b:min(b+batchSize, testSize)]
        bg_batch = testBg[b:min(b+batchSize, testSize)]
        
        # Count total words (excluding padding and start tokens)
        batch_words = sum(len(s)-1 for s in eng_batch) + sum(len(s)-1 for s in bg_batch)
        total_words += batch_words
        
        with torch.no_grad():
            # Get combined loss from the model
            loss = nmt(eng_batch, bg_batch)
            total_loss += loss.item() * batch_words
    
    nmt.train()  # Set model back to training mode
    return math.exp(total_loss/total_words)


if len(sys.argv)>1 and sys.argv[1] == 'prepare':
    def get_word2ind(sp_model):
        """Returns a dictionary mapping subwords to their indices."""
        word2ind = {sp_model.id_to_piece(i): i for i in range(sp_model.get_piece_size())}
        return word2ind

    trainCorpusBg, trainCorpusEng, devCorpusBg, devCorpusEng, sp_source, sp_target = utils.prepareDataBPE(sourceFileName, targetFileName, sourceDevFileName, targetDevFileName, bpe_Eng, bpe_Bg)
    word2indBg = get_word2ind(sp_target)
    word2indEng = get_word2ind(sp_source)

    pickle.dump((trainCorpusBg, trainCorpusEng, devCorpusBg, devCorpusEng), open(corpusFileName, 'wb'))
    pickle.dump((word2indEng, word2indBg), open(wordsFileName, 'wb'))

    print('Data prepared.')


if len(sys.argv)>1 and (sys.argv[1] == 'train' or sys.argv[1] == 'extratrain'):
    (trainCorpusBg, trainCorpusEng, devCorpusBg, devCorpusEng) = pickle.load(open(corpusFileName, 'rb'))

    word2indEng, word2indBg = pickle.load(open(wordsFileName, 'rb'))

    nmt = model.LanguageModel(emd_size, hidden_size, word2indEng, word2indBg, unkToken, padToken,
                        endToken, lstm_layers, dropout_encoder,
                        dropout_translator, dropaut_generator, dropout_attention).to(device)    
    optimizer = torch.optim.Adam(nmt.parameters(), lr=learning_rate)

    if sys.argv[1] == 'extratrain':
        nmt.load(modelFileName)
        (iter,bestPerplexity,learning_rate,osd) = torch.load(modelFileName + '.optim')
        optimizer.load_state_dict(osd)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    else:
        bestPerplexity = math.inf
        iter = 0

    idx = np.arange(len(trainCorpusBg), dtype='int32')
    nmt.train()
    beginTime = time.time()
    for epoch in range(maxEpochs):
        np.random.shuffle(idx)
        words = 0
        trainTime = time.time()
        for b in range(0, len(idx), batchSize):
			#############################################################################
			### Може да се наложи да се променя скоростта на спускане learning_rate в зависимост от итерацията
			#############################################################################
            iter += 1
            batchEng = [ trainCorpusEng[i] for i in idx[b:min(b+batchSize, len(idx))] ]
            batchBg = [ trainCorpusBg[i] for i in idx[b:min(b+batchSize, len(idx))] ]

            
            words += sum( len(s)-1 for s in batchEng ) + sum( len(s)-1 for s in batchBg )
            H = nmt(batchEng, batchBg)
            optimizer.zero_grad()
            H.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(nmt.parameters(), clip_grad)
            optimizer.step()
            if iter % log_every == 0:
                print("Iteration:",iter,"Epoch:",epoch+1,'/',maxEpochs,", Batch:",b//batchSize+1, '/', len(idx) // batchSize+1, ", loss: ",H.item(), "words/sec:",words / (time.time() - trainTime), "time elapsed:", (time.time() - beginTime) )
                trainTime = time.time()
                words = 0
                
            if iter % test_every == 0:
                nmt.eval()
                currentPerplexity = perplexity(nmt, devCorpusEng, devCorpusBg, batchSize)
                nmt.train()
                print('Current model perplexity: ',currentPerplexity)

                if currentPerplexity < bestPerplexity:
                    bestPerplexity = currentPerplexity
                    print('Saving new best model.')
                    nmt.save(modelFileName)
                    torch.save((iter,bestPerplexity,learning_rate,optimizer.state_dict()), modelFileName + '.optim')

    print('reached maximum number of epochs!')
    nmt.eval()
    currentPerplexity = perplexity(nmt, devCorpusEng, devCorpusBg, batchSize)
    print('Last model perplexity: ',currentPerplexity)
        
    if currentPerplexity < bestPerplexity:
        bestPerplexity = currentPerplexity
        print('Saving last model.')
        nmt.save(modelFileName)
        torch.save((iter,bestPerplexity,learning_rate,optimizer.state_dict()), modelFileName + '.optim')

if len(sys.argv)>3 and sys.argv[1] == 'perplexity':
    word2ind = pickle.load(open(wordsFileName, 'rb'))
    
    nmt = model.LanguageModel(emd_size, hidden_size, word2indEng, word2indBg, unkToken, padToken,
                        endToken, lstm_layers, dropout_encoder,
                        dropout_translator, dropaut_generator, dropout_attention).to(device)    
    nmt.load(modelFileName)
    
    sourceTest = utils.readCorpus(sys.argv[2])
    targetTest = utils.readCorpus(sys.argv[3])
    testEng = [ [startToken] + s + [endToken] for s in sourceTest]
    testBg = [ [startToken] + s + [endToken] for s in targetTest]

    nmt.eval()
    print('Model perplexity: ', perplexity(nmt, testEng, testBg, batchSize))

if len(sys.argv)>3 and sys.argv[1] == 'translate':
    word2ind = pickle.load(open(wordsFileName, 'rb'))
    words = list(word2ind)

    sourceTest = utils.readCorpus(sys.argv[2])
    test = [ [startToken] + s + [endToken] for s in sourceTest ]

    nmt = model.LanguageModel(emd_size, hidden_size, word2indEng, word2indBg, unkToken, padToken,
                        endToken, lstm_layers, dropout_encoder,
                        dropout_translator, dropaut_generator, dropout_attention).to(device)    
    nmt.load(modelFileName)

    nmt.eval()
    file = open(sys.argv[3],'w')
    pb = utils.progressBar()
    pb.start(len(test))
    for s in test:
        r=nmt.generate(s)
        file.write(' '.join(r)+"\n")
        pb.tick()
    pb.stop()

if len(sys.argv)>2 and sys.argv[1] == 'generate':
    word2indEng, word2indBg = pickle.load(open(wordsFileName, 'rb'))

    test = sys.argv[2]

    nmt = model.LanguageModel(emd_size, hidden_size, word2indEng, word2indBg, unkToken, padToken,
                        endToken, lstm_layers, dropout_encoder,
                        dropout_translator, dropaut_generator, dropout_attention).to(device)    
    nmt.load(modelFileName)

    nmt.eval()
    r=nmt.generate(test)
    print(' '.join(r)+"\n")

if len(sys.argv)>3 and sys.argv[1] == 'bleu':
    ref = [[s] for s in utils.readCorpus(sys.argv[2])]
    hyp = utils.readCorpus(sys.argv[3])

    bleu_score = corpus_bleu(ref, hyp)
    print('Corpus BLEU: ', (bleu_score * 100))
