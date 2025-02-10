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

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear((hidden_size * 2), hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, hidden_outputs):
        # hidden: (batch_size, hidden_size)
        # hidden_outputs: (batch_size, number, hidden_size)

        num = hidden_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, num, 1)  # (batch_size, curr_pos, hidden_size)

        energy = torch.tanh(self.attn(torch.cat((hidden, hidden_outputs), dim = 2)))  # (batch_size, curr_pos, hidden_size)

        attention = self.v(energy).squeeze(2)  # (batch_size, curr_pos)
		
        return torch.softmax(attention, dim=1)  # (batch_size, curr_pos)

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, dropout, attention):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.attention = attention
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.LSTM(hidden_size + emb_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear((hidden_size * 2) + emb_size, vocab_size)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell, hidden_outputs):
        # hidden (num_layers, batch_size, hidden_size)
        # cell (num_layers, batch_size, hidden_size)
        # hidden_outputs (batch_size, count, hidden_size)

        x = x.unsqueeze(0)  # (1, batch_size)

        e = self.dropout(self.embedding(x)) # (1, batch_size, embedding_size)

        hidden_attention = hidden[-1] if self.num_layers > 1 else hidden.squeeze(0)  # (batch_size, hidden_size)

        att = self.attention(hidden_attention, hidden_outputs)  # (batch_size, count)

        att = att.unsqueeze(1)  # (batch_size, 1, curr_pos)

        weighted = torch.bmm(att, hidden_outputs)  # (batch_size, 1, hidden_size)

        weighted = weighted.permute(1, 0, 2)  # (1, batch_size, hidden_size)

        rnn_input = torch.cat((e, weighted), dim=2)  # (1, batch_size, hidden_size + embedding_size)

        rnn_input = rnn_input.transpose(1, 0)

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))  # output: (batch_size, 1, hidden_size)
        output = output.transpose(1, 0)

        e = e.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc(torch.cat((output, weighted, e), dim = 1))

        #prediction = [batch_size, output_dim]

        return prediction, hidden.squeeze(0), cell.squeeze(0)

class LanguageModel(torch.nn.Module):
    def __init__(self, embed_size, hidden_size, word2ind, startToken, unkToken, padToken, endToken, transToken,
                  lstm_layers, dropout, dropout_translator, maxlen=1000):
        super(LanguageModel, self).__init__()
        self.startTokenIdx = word2ind[startToken]
        self.padTokenIdx = word2ind[padToken]
        self.endTokenIdx = word2ind[endToken]
        self.unkTokenIdx = word2ind[unkToken]
        self.transToken = word2ind[transToken]
        self.num_layers = lstm_layers
        self.hidden_size = hidden_size
        self.device = next(self.parameters()).device
        
        self.attention = Attention(hidden_size)
        self.decoder = Decoder(len(word2ind), embed_size, hidden_size, lstm_layers, dropout_translator, self.attention)

        self.dropout = nn.Dropout(dropout)
	
    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in source]
        return torch.tensor(sents_padded, dtype=torch.long, device=device)	# shape=(batch_size, seq_len)

    def save(self,fileName):
        torch.save(self.state_dict(), fileName)

    def load(self,fileName):
        self.load_state_dict(torch.load(fileName))

    def forward(self, source, teacher_forcing_ratio = 0.5):
        src = self.preparePaddedBatch(source)

        batch_size, src_len = src.shape
        vocab_size = self.decoder.vocab_size

        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, src_len, vocab_size).to(src.device)

        #first input to the decoder is the <s> tokens
        input = src[:, 0]

        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        hidden_outputs = torch.zeros(batch_size, 1, self.hidden_size)
        

        for t in range(1, src_len):
            #insert input token embedding, previous hidden state, all encoder hidden states
            #  and mask
            #receive output tensor (predictions) and new hidden state
            output, hidden, cell = self.decoder(input, hidden, cell, hidden_outputs)

            hidden_attention = hidden[-1] if self.num_layers > 1 else hidden.squeeze(0)  # (batch_size, hidden_size)

            hidden_outputs = torch.cat((hidden_outputs, hidden_attention.unsqueeze(1)), dim = 1)

            #place predictions in a tensor holding predictions for each token
            outputs[:, t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            #get the highest predicted token from our predictions
            top1 = output.argmax(1)

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = src[:, t] if teacher_force else top1

        #outputs = [batch_size, src_len, vocab_size]

        return self.loss(outputs.view(-1, vocab_size), src.view(-1))
        
    def generate(self, prefix, limit=1000):
        return result
