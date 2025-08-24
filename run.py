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
import argparse
import csv

from nltk.translate.bleu_score import corpus_bleu

import utils
import model
import parameters as params

# Token definitions
startToken = '<s>'
endToken = '</s>'
unkToken = '<unk>'
padToken = '<pad>'
transToken = '<trans>'

def perplexity(nmt, testEng, testBg, batchSize):
    testSize = min(len(testEng), len(testBg))
    total_loss = 0.0
    total_words = 0
    
    nmt.eval()
    
    for b in range(0, testSize, batchSize):
        eng_batch = testEng[b:min(b+batchSize, testSize)]
        bg_batch = testBg[b:min(b+batchSize, testSize)]
        
        batch_words = sum(len(s)-1 for s in eng_batch) + sum(len(s)-1 for s in bg_batch)
        total_words += batch_words
        
        with torch.no_grad():
            loss = nmt(eng_batch, bg_batch)
            total_loss += loss.item() * batch_words
    
    nmt.train()
    if total_words == 0:
        return float('inf')
    return math.exp(total_loss/total_words)

def create_model(args, word2indEng, word2indBg):
    """Creates and returns the NMT model."""
    return model.LanguageModel(
        args.embed_size, args.hidden_size, word2indEng, word2indBg, startToken, unkToken, padToken,
        endToken, transToken, args.lstm_layers, args.dropout_encoder,
        args.dropout_translator, args.dropout_generator
    ).to(args.device)

def prepare(args):
    def get_word2ind(sp_model):
        """Returns a dictionary mapping subwords to their indices."""
        return {sp_model.id_to_piece(i): i for i in range(sp_model.get_piece_size())}

    trainCorpusBg, trainCorpusEng, devCorpusBg, devCorpusEng, sp_source, sp_target = utils.prepareDataBPE(
        args.source_file, args.target_file, args.source_dev_file, args.target_dev_file, args.bpe_eng, args.bpe_bg
    )
    word2indBg = get_word2ind(sp_target)
    word2indEng = get_word2ind(sp_source)

    pickle.dump((trainCorpusBg, trainCorpusEng, devCorpusBg, devCorpusEng), open(args.corpus_file, 'wb'))
    pickle.dump((word2indEng, word2indBg), open(args.words_file, 'wb'))

    print('Data prepared and saved to', args.corpus_file, 'and', args.words_file)

def train(args):
    (trainCorpusBg, trainCorpusEng, devCorpusBg, devCorpusEng) = pickle.load(open(args.corpus_file, 'rb'))
    (word2indEng, word2indBg) = pickle.load(open(args.words_file, 'rb'))

    nmt = create_model(args, word2indEng, word2indBg)
    nmt.summary()

    optimizer = torch.optim.Adam(nmt.parameters(), lr=args.lr)

    if args.resume:
        nmt.load(args.model_file, map_location=args.device)
        (iter_num, bestPerplexity, _, osd) = torch.load(args.model_file + '.optim')
        optimizer.load_state_dict(osd)
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        print(f"Resuming training from iteration {iter_num} with best perplexity {bestPerplexity}")
    else:
        with open(args.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration', 'Epoch', 'Batch', 'Loss', 'Words/sec', 'Time elapsed', 'Best Perplexity'])
        bestPerplexity = math.inf
        iter_num = 0

    idx = np.arange(len(trainCorpusBg), dtype='int32')
    nmt.train()
    beginTime = time.time()
    for epoch in range(args.max_epochs):
        np.random.shuffle(idx)
        words = 0
        trainTime = time.time()
        for b in range(0, len(idx), args.batch_size):
            iter_num += 1
            batchEng = [trainCorpusEng[i] for i in idx[b:min(b + args.batch_size, len(idx))]]
            batchBg = [trainCorpusBg[i] for i in idx[b:min(b + args.batch_size, len(idx))]]

            words += sum(len(s) - 1 for s in batchEng) + sum(len(s) - 1 for s in batchBg)
            H = nmt(batchEng, batchBg)
            optimizer.zero_grad()
            H.backward()
            torch.nn.utils.clip_grad_norm_(nmt.parameters(), args.clip_grad)
            optimizer.step()

            if iter_num % args.log_every == 0:
                print(f"Iteration: {iter_num}, Epoch: {epoch+1}/{args.max_epochs}, Batch: {b//args.batch_size+1}/{len(idx)//args.batch_size+1}, loss: {H.item():.4f}, words/sec: {words/(time.time() - trainTime):.2f}, time elapsed: {(time.time() - beginTime):.2f}")
                with open(args.log_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([iter_num, epoch + 1, b // args.batch_size + 1, H.item(), words / (time.time() - trainTime), (time.time() - beginTime), bestPerplexity])
                trainTime = time.time()
                words = 0

            if iter_num % args.test_every == 0:
                nmt.eval()
                currentPerplexity = perplexity(nmt, devCorpusEng, devCorpusBg, args.batch_size)
                nmt.train()
                print('Current model perplexity:', currentPerplexity)

                if currentPerplexity < bestPerplexity:
                    bestPerplexity = currentPerplexity
                    print('Saving new best model.')
                    nmt.save(args.model_file)
                    torch.save((iter_num, bestPerplexity, args.lr, optimizer.state_dict()), args.model_file + '.optim')

    print('Reached maximum number of epochs!')
    nmt.eval()
    currentPerplexity = perplexity(nmt, devCorpusEng, devCorpusBg, args.batch_size)
    print('Last model perplexity:', currentPerplexity)

    if currentPerplexity < bestPerplexity:
        print('Saving last model.')
        nmt.save(args.model_file)
        torch.save((iter_num, bestPerplexity, args.lr, optimizer.state_dict()), args.model_file + '.optim')

def translate(args):
    (_, word2indBg) = pickle.load(open(args.words_file, 'rb'))
    (word2indEng, _) = pickle.load(open(args.words_file, 'rb'))

    nmt = create_model(args, word2indEng, word2indBg)
    nmt.load(args.model_file, map_location=args.device)
    nmt.eval()

    sourceTest = utils.readCorpus(args.source_test_file)
    test = [[startToken] + s + [endToken] for s in sourceTest]

    with open(args.output_file, 'w') as file:
        pb = utils.progressBar()
        pb.start(len(test))
        for s in test:
            r = nmt.generate(s)
            file.write(' '.join(r) + "\n")
            pb.tick()
        pb.stop()
    print("Translated sentences saved to", args.output_file)

def generate(args):
    (word2indEng, word2indBg) = pickle.load(open(args.words_file, 'rb'))

    nmt = create_model(args, word2indEng, word2indBg)
    nmt.load(args.model_file, map_location=args.device)
    nmt.eval()

    r = nmt.generate(args.prefix)
    print(' '.join(r) + "\n")

def bleu_score(args):
    ref = [[s] for s in utils.readCorpus(args.reference_file)]
    hyp = utils.readCorpus(args.hypothesis_file)

    bleu = corpus_bleu(ref, hyp)
    print('Corpus BLEU:', (bleu * 100))

def main():
    parser = argparse.ArgumentParser(description='NMT Model using PyTorch')

    # Common arguments
    parser.add_argument('--device', default=params.device, help='Device to run on (e.g., "cuda:0" or "cpu")')
    parser.add_argument('--model-file', default=params.modelFileName, help='Path to save/load the model')
    parser.add_argument('--words-file', default=params.wordsFileName, help='Path to vocab file')

    subparsers = parser.add_subparsers(dest='command', required=True)

    # Prepare command
    parser_prepare = subparsers.add_parser('prepare', help='Prepare data and build vocabulary')
    parser_prepare.add_argument('--source-file', default=params.sourceFileName)
    parser_prepare.add_argument('--target-file', default=params.targetFileName)
    parser_prepare.add_argument('--source-dev-file', default=params.sourceDevFileName)
    parser_prepare.add_argument('--target-dev-file', default=params.targetDevFileName)
    parser_prepare.add_argument('--bpe-eng', default=params.bpe_Eng)
    parser_prepare.add_argument('--bpe-bg', default=params.bpe_Bg)
    parser_prepare.add_argument('--corpus-file', default=params.corpusFileName, help='Path to save the processed corpus')
    parser_prepare.set_defaults(func=prepare)

    # Create a parent parser for arguments shared by train, translate, and generate
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument('--embed-size', type=int, default=params.emd_size)
    model_parser.add_argument('--hidden-size', type=int, default=params.hidden_size)
    model_parser.add_argument('--lstm-layers', type=int, default=params.lstm_layers)
    model_parser.add_argument('--dropout-encoder', type=float, default=params.dropout_encoder)
    model_parser.add_argument('--dropout-translator', type=float, default=params.dropout_translator)
    # Correcting typo from parameters.py
    model_parser.add_argument('--dropout-generator', type=float, default=params.dropaut_generator)

    # Train command
    parser_train = subparsers.add_parser('train', help='Train the model', parents=[model_parser])
    parser_train.add_argument('--corpus-file', default=params.corpusFileName)
    parser_train.add_argument('--log-file', default=params.log_filename)
    parser_train.add_argument('--lr', type=float, default=params.learning_rate)
    parser_train.add_argument('--batch-size', type=int, default=params.batchSize)
    parser_train.add_argument('--clip-grad', type=float, default=params.clip_grad)
    parser_train.add_argument('--max-epochs', type=int, default=params.maxEpochs)
    parser_train.add_argument('--log-every', type=int, default=params.log_every)
    parser_train.add_argument('--test-every', type=int, default=params.test_every)
    parser_train.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser_train.set_defaults(func=train)

    # Translate command
    parser_translate = subparsers.add_parser('translate', help='Translate a file', parents=[model_parser])
    parser_translate.add_argument('source_test_file', help='Path to the source file to translate')
    parser_translate.add_argument('output_file', help='Path to save the translated output')
    parser_translate.set_defaults(func=translate)

    # Generate command
    parser_generate = subparsers.add_parser('generate', help='Generate text from a prefix', parents=[model_parser])
    parser_generate.add_argument('prefix', help='The prefix to start generation from')
    parser_generate.set_defaults(func=generate)

    # BLEU score command
    parser_bleu = subparsers.add_parser('bleu', help='Calculate BLEU score')
    parser_bleu.add_argument('reference_file', help='Path to the reference translation file')
    parser_bleu.add_argument('hypothesis_file', help='Path to the hypothesis translation file')
    parser_bleu.set_defaults(func=bleu_score)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
