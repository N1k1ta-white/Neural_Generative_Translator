#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2024/2025
#############################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import random
import torch
from torch import nn

import sentencepiece as spm

# Additive Attention
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear((hidden_size * 2) + hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: (batch_size, hidden_size)
        # encoder_outputs: (batch_size, seq_len, hidden_size * 2)

        seq_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_size)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))  # (batch_size, seq_len, hidden_size)

        attention = self.v(energy).squeeze(2)  # (batch_size, seq_len)

        attention = attention.masked_fill(mask == 0, -1e10)

        return torch.softmax(attention, dim=1)  # (batch_size, seq_len)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout, padTokenIdx):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padTokenIdx)
        self.dropout = nn.Dropout(dropout)
        # GRU layer (bidirectional)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.hidden_size = hidden_size
        self.padTokenIdx = padTokenIdx
        self.num_layers = num_layers
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def transformStates(self, hidden, cell):
        combined_hidden = []
        combined_cell = []
        for layer in range(self.num_layers):
            h_forward = hidden[2 * layer, :, :]    # Forward direction
            h_backward = hidden[2 * layer + 1, :, :]  # Backward direction
            combined_h = torch.tanh(self.fc(torch.cat((h_forward, h_backward), dim=1)))
            combined_hidden.append(combined_h)

            c_forward = cell[2 * layer, :, :]
            c_backward = cell[2 * layer + 1, :, :]
            combined_c = torch.tanh(self.fc(torch.cat((c_forward, c_backward), dim=1)))
            combined_cell.append(combined_c)

        # Stack to [num_layers, batch_size, hidden_size]
        return (torch.stack(combined_hidden, dim=0), torch.stack(combined_cell, dim=0))

    def executeThrowRnn(self, X, source_lengths):
        E = self.dropout(self.embedding(X)) # (batch_size, seq_len, embedding_size)

        packed = torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths, batch_first = True, enforce_sorted=False)

        outputPacked, (hidden, cell) = self.rnn(packed)  # outputPacked: (batch_size, seq_len, 2 * hidden_size)

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)  # (batch_size, seq_len, 2 * hidden_size)

        return output, (hidden, cell)

    def forward(self, X, source_lengths):
        # source: (batch_size, seq_len)

        output, (hidden, cell) = self.executeThrowRnn(X, source_lengths)
        # hidden (2 * num_layers, batch_size, hidden_size)

        (hidden, cell) = self.transformStates(hidden, cell)
        return output, (hidden, cell)

    def forward_step(self, x, hidden, cell):
        e = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.rnn(e, (hidden, cell))
        return output, (hidden, cell)

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, dropout, attention):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.attention = attention
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.LSTM((hidden_size * 2) + emb_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear((hidden_size * 2) + hidden_size + emb_size, vocab_size)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell, encoder_outputs, mask):
        # hidden (num_layers, batch_size, hidden_size)
        # cell (num_layers, batch_size, hidden_size)
        # encoders_outputs (batch_size, seq_len - 1, hidden_size * 2)
        # mask (seq_len ,batch_size)

        x = x.unsqueeze(0)  # (1, batch_size)

        e = self.dropout(self.embedding(x)) # (1, batch_size, embedding_size)

        hidden_attention = hidden[-1] if self.num_layers > 1 else hidden.squeeze(0)  # (batch_size, hidden_size)

        att = self.attention(hidden_attention, encoder_outputs, mask)  # (batch_size, seq_len)

        att = att.unsqueeze(1)  # (batch_size, 1, seq_len)

        weighted = torch.bmm(att, encoder_outputs)  # (batch_size, 1, hidden_size * 2)

        weighted = weighted.permute(1, 0, 2)  # (1, batch_size, hidden_size * 2)

        rnn_input = torch.cat((e, weighted), dim=2)  # (1, batch_size, hidden_size * 2 + embedding_size)

        rnn_input = rnn_input.transpose(1, 0)

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))  # output: (batch_size, 1, hidden_size)
        output = output.transpose(1, 0)

        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]

        e = e.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc(torch.cat((output, weighted, e), dim = 1))

        #prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0), cell.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, decoder, src_pad_idx):
        super(Seq2Seq, self).__init__()
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.loss = nn.CrossEntropyLoss(ignore_index = src_pad_idx)

    def forward(self, trg, encoder_outputs, hidden, cell, mask, teacher_forcing_ratio = 0.3):
        #src = [batch_size, src_len]
        #src_len = [batch_size]
        #trg = [batch_size, trg_len]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.vocab_size

        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(trg.device)

        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer

        #first input to the decoder is the <s> tokens
        input = trg[:, 0]

        for t in range(1, trg_len):
            #insert input token embedding, previous hidden state, all encoder hidden states
            #  and mask
            #receive output tensor (predictions) and new hidden state
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs, mask)

            #place predictions in a tensor holding predictions for each token
            outputs[:, t] = output

            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            #get the highest predicted token from our predictions
            top1 = output.argmax(1)

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[:, t] if teacher_force else top1

        return self.loss(outputs.view(-1, trg_vocab_size), trg.view(-1))

class TextGenerator(nn.Module):
    def __init__(self, decoder, src_pad_idx):
        super(TextGenerator, self).__init__()
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.loss = nn.CrossEntropyLoss(ignore_index = src_pad_idx)

    def create_temp_mask(self, mask, t):
        new_mask = mask.clone()
        new_mask[:, t:] = False
        return new_mask

    def forward(self, src, encoder_outputs, hidden, cell, mask):
        # src : (batch_size, src_len)
        # src_len : (batch_size)

        batch_size, src_len = src.shape
        vocab_size = self.decoder.vocab_size

        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, src_len, vocab_size).to(src.device)

        #first input to the decoder is the <s> tokens
        input = src[:, 0]

        for t in range(1, src_len):
            temp_mask = self.create_temp_mask(mask, t - 1)
            #insert input token embedding, previous hidden state, all encoder hidden states
            #  and mask
            #receive output tensor (predictions) and new hidden state
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs, temp_mask)

            #place predictions in a tensor holding predictions for each token
            outputs[:, t] = output
            input = src[:, t]

        #outputs = [batch_size, src_len, vocab_size]

        return self.loss(outputs.view(-1, vocab_size), src.view(-1))

class LanguageModel(torch.nn.Module):
    def __init__(self, embed_size, hidden_size, word2indEng, word2indBg, startToken, unkToken, padToken, endToken, transToken,
                  lstm_layers, dropout_encoder, dropout_translator, dropaut_generator, maxlen=1000):
        super(LanguageModel, self).__init__()
        self.word2indBg = word2indBg
        self.word2indEng = word2indEng
        self.startTokenIdx = word2indBg[startToken]
        self.padTokenIdx = word2indBg[padToken]
        self.endTokenIdx = word2indBg[endToken]
        self.unkTokenIdx = word2indBg[unkToken]
        self.transToken = transToken

        self.encoder = Encoder(len(word2indEng), embed_size, hidden_size, lstm_layers, dropout_encoder, self.padTokenIdx)
        self.attention = Attention(hidden_size)
        self.DecoderTransltor = Decoder(len(word2indBg), embed_size, hidden_size, lstm_layers, dropout_translator, self.attention)
        self.DecoderGenerator = Decoder(len(word2indEng), embed_size, hidden_size, lstm_layers, dropaut_generator, self.attention)

        self.Seq2Seq = Seq2Seq(self.DecoderTransltor, self.padTokenIdx)
        self.TextGenerator = TextGenerator(self.DecoderGenerator, self.padTokenIdx)

        self.dropout = nn.Dropout(dropout_encoder)

        # task weights (learnable parameters)
        device = next(self.parameters()).device
        self.task_weights = nn.Parameter(torch.ones(2).to(device))
        self.prev_losses = [torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)]

    def preparePaddedBatch(self, sourceEng, sourceBg):
        device = next(self.parameters()).device
        m = max(len(s) for s in sourceEng)
        m = max(m, max(len(s) for s in sourceBg))
        sents_paddedEng = [s+(m-len(s))*[self.padTokenIdx] for s in sourceEng]
        sents_paddedBg = [s + (m - len(s))*[self.padTokenIdx] for s in sourceBg]
        return torch.tensor(sents_paddedEng, dtype=torch.long, device=device), torch.tensor(sents_paddedBg, dtype=torch.long, device=device)	# shape=(batch_size, seq_len)

    def save(self,fileName):
        torch.save(self.state_dict(), fileName)

    def load(self,fileName, map_location = torch.device("cuda:0")):
        self.load_state_dict(torch.load(fileName, map_location))

    def create_mask(self, src):
        mask = (src != self.padTokenIdx)
        return mask[:, 1:]         #mask = [batch size, src len]

    def forward(self, engBatch, bgBatch):
        engBatchPadded, bgBatchPadded = self.preparePaddedBatch(engBatch, bgBatch)

        engLength = [len(s) - 1 for s in engBatch]

        encoder_outputs, (hidden, cell) = self.encoder(engBatchPadded, engLength)
        encoder_outputs.transpose_(0, 1)  # (seq_len, batch_size, 2 * hidden_size)

        encoder_outputs = self.dropout(encoder_outputs)

        mask = self.create_mask(engBatchPadded)

        hiddenGen = torch.zeros_like(hidden)
        cellGen= torch.zeros_like(cell)

        # Current losses
        L1 = self.TextGenerator(engBatchPadded, encoder_outputs, hiddenGen, cellGen, mask)  # Generation loss
        L2 = self.Seq2Seq(bgBatchPadded, encoder_outputs, hidden, cell, mask)  # Translation loss

        # Calculate relative inverse training rates
        r1 = (L1 / self.prev_losses[0].detach()).detach()
        r2 = (L2 / self.prev_losses[1].detach()).detach()
        r_mean = torch.mean(torch.tensor([r1, r2]))

        # Update weights using GradNorm
        w1 = self.task_weights[0] * (r1 / r_mean)
        w2 = self.task_weights[1] * (r2 / r_mean)

        # Normalize weights
        weights = torch.softmax(torch.stack([w1, w2]), dim=0)
        w1, w2 = weights[0], weights[1]

        # Update task weights
        self.task_weights.data = torch.tensor([w1, w2]).to(self.task_weights.device)

        # Store current losses for next iteration
        self.prev_losses = [L1.detach(), L2.detach()]

        # Return weighted sum of losses
        return w1 * L1 + w2 * L2

    def isCompleteSentence(self, prefix):
        return self.transToken in prefix

    def generate(self, prefix, temperature = 0.5, max_len=1000):
        spEng = spm.SentencePieceProcessor(model_file=bpe_Eng)

        device = next(self.parameters()).device

        tokens = [self.startTokenIdx] + spEng.encode(prefix, out_type=int)

        src_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        src_len = torch.tensor([len(tokens) - 1], dtype=torch.long, device='cpu')

        generated = []
        self.eval()

        with torch.no_grad():
            encoder_outputs, (hiddenEn, cellEn) = self.encoder.executeThrowRnn(src_tensor, src_len)
            encoder_outputs = encoder_outputs.transpose(1, 0)
            mask = self.create_mask(src_tensor)

            num_layers, batch, size = hiddenEn.shape

            hidden = torch.zeros(num_layers // 2, batch, size, device=device)
            cell = torch.zeros(num_layers // 2, batch, size, device=device)

            input = src_tensor[:, 0].to(device)

            for t in range(1, len(tokens)):
                temp_mask = self.TextGenerator.create_temp_mask(mask, t - 1)
                output, hidden, cell = self.TextGenerator.decoder(input, hidden, cell,
                                                                   encoder_outputs, temp_mask)
                input = src_tensor[:, t]
            
            for t in range(1, max_len):
                #insert input token embedding, previous hidden state, all encoder hidden states
                #  and mask
                #receive output tensor (predictions) and new hidden state
                encoder_output, (hiddenEn, cellEn) = self.encoder.forward_step(input.unsqueeze(1), hiddenEn, cellEn)
                encoder_outputs = torch.cat((encoder_outputs, encoder_output), dim = 1)
                mask = torch.cat((mask, torch.tensor([[True]], device=device)), dim = 1)
                output, hidden, cell = self.TextGenerator.decoder(input, hidden, cell, encoder_outputs, mask)

                prediction = torch.softmax(output / temperature, dim=-1)

                input = torch.multinomial(prediction, 1)
                input = input.squeeze(1)
                
                  # Stop if END token is generated
                if next_token == self.endTokenIdx:
                    break

                # Get next token
                next_token = input.item()
                generated.append(next_token)

                

        self.train()
        return prefix + spEng.decode(generated)

    def translate(self, prefix, temperature = 0.5, max_len = 1000):
        spEng = spm.SentencePieceProcessor(model_file=bpe_Eng)
        spBg = spm.SentencePieceProcessor(model_file=bpe_Bg)
        device = next(self.parameters()).device

        tokens = [self.startTokenIdx] + spEng.encode(prefix, out_type=int) + [self.endTokenIdx]

        src_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        src_len = torch.tensor([len(tokens) - 1], dtype=torch.long, device=device)

        generated = []
        self.eval()
        with torch.no_grad():
            encoder_outputs, (hidden, cell) = self.encoder(src_tensor, src_len)
            encoder_outputs = encoder_outputs.transpose(1, 0)
            mask = self.create_mask(src_tensor)

            input = torch.tensor([self.startTokenIdx], dtype=torch.long, device=device)

            for t in range(1, max_len):
                #insert input token embedding, previous hidden state, all encoder hidden states
                #  and mask
                #receive output tensor (predictions) and new hidden state
                output, hidden, cell = self.Seq2Seq.decoder(input, hidden, cell, encoder_outputs, mask)

                prediction = torch.softmax(output / temperature, dim=-1)

                input = torch.multinomial(prediction, 1)
                input = input.squeeze(1)

                # Get next token
                next_token = input.item()
                generated.append(next_token)

                # Stop if END token is generated
                if next_token == self.endTokenIdx:
                    break

        self.train()
        return spBg.decode(generated)