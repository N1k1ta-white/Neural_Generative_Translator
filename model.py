#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2024/2025
#############################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import torch
from torch import nn

# Additive Attention
class Attention(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: (batch_size, hidden_size)
        # encoder_outputs: (batch_size, seq_len, hidden_size)

        seq_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, hidden_size)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))  # (batch_size, seq_len, hidden_size)

        attention = self.dropout(self.v(energy).squeeze(2))  # (batch_size, seq_len)

        if mask is not None:
            mask = mask.to(attention.device)
            attention = attention.masked_fill(mask.unsqueeze(0).unsqueeze(1), -1e9)

        att = torch.softmax(attention, dim=1)  # (batch_size, seq_len)
        return att

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout, padTokenIdx):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padTokenIdx)
        self.dropout = nn.Dropout(dropout)
        # LSTM layer (bidirectional)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.projection = nn.Linear(2 * hidden_size, hidden_size)
        self.hidden_size = hidden_size
        self.padTokenIdx = padTokenIdx
        self.num_layers = num_layers

    def forward(self, X, source_lengths):
        # source: (batch_size, seq_len)

        batch_size, seq_len = X.shape
        E = self.embedding(X) # (batch_size, seq_len, embedding_size)

        packed = torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths, batch_first = True, enforce_sorted=False)

        h0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_size).to(E.device)
        c0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_size).to(E.device)

        outputPacked, (hidden, cell) = self.rnn(packed, (h0, c0))  # output: (batch_size, seq_len, 2 * hidden_size)

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)  # (batch_size, seq_len, 2 * hidden_size)

        t = self.projection(output)

        return t, hidden, cell

class DecoderGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout, dropout_attention, padTokenIdx, mask):
        super(DecoderGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.attention = Attention(hidden_size, dropout_attention)  # Use Attention
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, vocab_size)
        self.mask = mask
        self.hidden_size = hidden_size
        self.padTokenIdx = padTokenIdx
        self.num_layers = num_layers

    def forward(self, X, encoderOutputs):
        # src and trg is array not tensor
        # src: (batch_size, src_len)  - Input sequence

        E = self.embedding(X[:, :-1]) # (batch_size, src_len, embedding_size)

        batch_size = X.shape[0]
        seq_len = X.shape[1]

        # packed = torch.nn.utils.rnn.pack_padded_sequence(E, [len(s) for s in src], enforce_sorted=False)

        hidden = torch.randn(self.num_layers, batch_size, self.hidden_size).to(X.device) * 0.01
        cell = torch.randn(self.num_layers, batch_size, self.hidden_size).to(X.device) * 0.01

        totalLoss = 0

        for t in range(seq_len - 1):
            # packed = torch.nn.utils.rnn.pack_sequence([E[:, t, :] for t in range(seq_len)], enforce_sorted=False)
            input_step = E[:, t].unsqueeze(1)  # (batch_size, 1, embedding_size)

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

            output, (hidden, cell) = self.rnn(input_step, (hidden, cell))

            hidden_attention = hidden[-1] if self.num_layers > 1 else hidden.squeeze(0)  # (batch_size, hidden_size)

            att = self.attention(hidden_attention, encoderOutputs, self.mask[t, :seq_len - 1])  # (batch_size, seq_len)
            att = att.transpose(0, 1)  # (batch_size, 1, seq_len)

            weighted = torch.bmm(att, encoderOutputs)  # (batch_size, 1, hidden_size)

            # Prediction
            output = output.squeeze(1)  # (batch_size, hidden_size)
            weighted = weighted.squeeze(1)  # (batch_size, hidden_size)
            # output = torch.nn.utils.rnn.pad_packed_sequence(output)
            output = self.fc(torch.cat((output, weighted), dim=1)) # (batch_size, vocab_size)

            totalLoss += nn.functional.cross_entropy(output, X[:, t + 1], ignore_index=self.padTokenIdx)

        return totalLoss / seq_len - 1


class DecoderTranslator(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout, dropout_attention, padTokenIdx):
        super(DecoderTranslator, self).__init__()
        self.vocab_size = vocab_size
        self.attention = Attention(hidden_size, dropout_attention)  # Use Attention
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, vocab_size)
        self.padTokenIdx = padTokenIdx
        self.hidden_size = hidden_size

    def forward(self, X, encoderOutputs):

        E = self.embedding(X[:, :-1]) # (batch_size, src_len, embedding_size)

        batch_size, seq_len = X.shape

        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(X.device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(X.device)

        totalLoss = 0

        for t in range(seq_len - 1):
            input_step = E[:, t].unsqueeze(1)  # (batch_size, 1, embedding_size) - Current input

            output, (hidden, cell) = self.rnn(input_step, (hidden, cell))
            # output: (batch_size, 1, hidden_size)

            hidden_attention = hidden[-1] if self.num_layers > 1 else hidden.squeeze(0)  # (batch_size, hidden_size)

            # Attention
            att = self.attention(hidden_attention, encoderOutputs)  # (batch_size, seq_len)

            att = att.unsqueeze(1)  # (batch_size, 1, seq_len)

            weighted = torch.bmm(att, encoderOutputs)  # (batch_size, 1, hidden_size)

            # Prediction
            output = output.squeeze(1)  # (batch_size, hidden_size)

            weighted = weighted.squeeze(1)  # (batch_size, hidden_size)
            prediction = self.fc(torch.cat((output, weighted), dim=1)) # (batch_size, vocab_size)

            totalLoss += nn.functional.cross_entropy(prediction, X[:, t + 1], ignore_index=self.padTokenIdx)

        return totalLoss / seq_len - 1

class LanguageModel(torch.nn.Module):
    def __init__(self, embed_size, hidden_size, word2ind, unkToken, padToken, endToken, transToken,
                  lstm_layers, dropout_encoder, dropout_translator, dropaut_generator, 
                  dropout_attention, maxlen=1000):
        super(LanguageModel, self).__init__()
        self.word2ind = word2ind
        self.padTokenIdx = word2ind[padToken]
        self.endTokenIdx = word2ind[endToken]
        self.transTokenIdx = word2ind[transToken]
        self.unkTokenIdx = word2ind[unkToken]

        pos = torch.arange(maxlen)
        self.mask = pos.unsqueeze(0) > pos.unsqueeze(1)

        self.encoder = Encoder(len(word2ind), embed_size, hidden_size, lstm_layers, dropout_encoder, self.padTokenIdx)
        self.transltor = DecoderTranslator(len(word2ind), embed_size, hidden_size, lstm_layers, dropout_translator, dropout_attention, self.padTokenIdx)
        self.generator = DecoderGenerator(len(word2ind), embed_size, hidden_size, lstm_layers, dropaut_generator, dropout_attention, self.padTokenIdx, self.mask)
        self.lossFn = nn.CrossEntropyLoss(ignore_index = self.padTokenIdx)
        self.dropout = nn.Dropout(dropout_encoder)

        # task weights (learnable parameters)
        self.task_weights = nn.Parameter(torch.ones(2).to(next(self.parameters()).device))
        self.prev_losses = [1.0, 1.0]  # Initialize with 1.0 to avoid division by zero

    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents_padded = [s+(m-len(s))*[self.padTokenIdx] for s in source]
        return torch.tensor(sents_padded, dtype=torch.long, device=device)	# shape=(batch_size, seq_len)

    def save(self,fileName):
        torch.save(self.state_dict(), fileName)

    def load(self,fileName):
        self.load_state_dict(torch.load(fileName))

    def forward(self, engBatch, bgBatch):
        engBatchPadded = self.preparePaddedBatch(engBatch)
        bgBatchPadded = self.preparePaddedBatch(bgBatch)

        engLength = [len(s) - 1 for s in engBatch]

        encoderOutputs, _, _ = self.encoder(engBatchPadded, engLength)
        encoderOutputs.transpose_(0, 1)  # (seq_len, batch_size, 2 * hidden_size)

        encoderOutputs = self.dropout(encoderOutputs)

        # Current losses
        L1 = self.generator(engBatchPadded, encoderOutputs)  # Generation loss
        L2 = self.transltor(bgBatchPadded, encoderOutputs)  # Translation loss

        # Calculate relative inverse training rates
        r1 = L1.item() / self.prev_losses[0]
        r2 = L2.item() / self.prev_losses[1]
        r_mean = torch.mean(torch.tensor([r1, r2]))

        # Update weights using GradNorm
        w1 = self.task_weights[0] * (r1 / r_mean)
        w2 = self.task_weights[1] * (r2 / r_mean)

        # Normalize weights
        total = w1 + w2
        w1 = w1 / total
        w2 = w2 / total

        # Update task weights
        self.task_weights.data = torch.tensor([w1, w2]).to(self.task_weights.device)

        # Store current losses for next iteration
        self.prev_losses = [L1.item(), L2.item()]

        # Return weighted sum of losses
        return w1 * L1 + w2 * L2

    def generate(self, prefix, limit=1000):
        result = None
        return result