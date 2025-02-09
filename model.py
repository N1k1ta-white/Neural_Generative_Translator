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

import sentencepiece as spm

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

        batch_size, _ = X.shape
        E = self.embedding(X) # (batch_size, seq_len, embedding_size)

        h0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_size).to(E.device)
        c0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_size).to(E.device)

        packed = torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths, batch_first = True, enforce_sorted=False)

        outputPacked, (hidden, cell) = self.rnn(packed, (h0, c0))  # output: (batch_size, seq_len, 2 * hidden_size)

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)  # (batch_size, seq_len, 2 * hidden_size)

        t = self.projection(output)

        return t, hidden, cell

    def forwardStep(self, X, hidden, cell):
        # X: (batch_size, 1)
        E = self.embedding(X)
        output, (hidden, cell) = self.rnn(E, (hidden, cell))
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
        # X: (batch_size, src_len)  - Input sequence

        E = self.embedding(X[:, :-1]) # (batch_size, src_len, embedding_size)

        batch_size = X.shape[0]
        seq_len = X.shape[1]

        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(X.device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(X.device)

        totalLoss = 0

        for t in range(seq_len - 1):
            input_step = E[:, t].unsqueeze(1)  # (batch_size, 1, embedding_size)

            output, (hidden, cell) = self.rnn(input_step, (hidden, cell))

            hidden_attention = hidden[-1] if self.num_layers > 1 else hidden.squeeze(0)  # (batch_size, hidden_size)

            att = self.attention(hidden_attention, encoderOutputs, self.mask[t - 1, :seq_len - 1])  # (batch_size, seq_len)
            att = att.transpose(0, 1)  # (batch_size, 1, seq_len)

            weighted = torch.bmm(att, encoderOutputs)  # (batch_size, 1, hidden_size)

            # Prediction
            output = output.squeeze(1)  # (batch_size, hidden_size)
            weighted = weighted.squeeze(1)  # (batch_size, hidden_size)
            output = self.fc(torch.cat((output, weighted), dim=1)) # (batch_size, vocab_size)

            totalLoss += nn.functional.cross_entropy(output, X[:, t + 1], ignore_index=self.padTokenIdx)

        return totalLoss / (seq_len - 1)

class DecoderTranslator(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout, dropout_attention, padTokenIdx):
        super(DecoderTranslator, self).__init__()
        self.vocab_size = vocab_size
        self.attention = Attention(hidden_size, dropout_attention)
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, vocab_size)
        self.padTokenIdx = padTokenIdx
        self.hidden_size = hidden_size

    def _bridge_state(self, enc_state, num_layers, hidden_size):
        """
        Convert a bidirectional encoder state into a unidirectional decoder initial state.
        
        Args:
            enc_state (torch.Tensor): Encoder state with shape (2*num_layers, batch_size, hidden_size).
            num_layers (int): Number of layers in the decoder.
            hidden_size (int): Hidden size of the decoder.
        
        Returns:
            torch.Tensor: Bridged state with shape (num_layers, batch_size, hidden_size).
        """
        # Reshape from (2*num_layers, batch_size, hidden_size)
        # to (num_layers, 2, batch_size, hidden_size)
        enc_state = enc_state.view(num_layers, 2, -1, hidden_size)
        # Average over the two directions (dim=1)
        return enc_state.mean(dim=1)


    def forward(self, X, encoderOutputs, hiddeEn, cellEn):

        E = self.embedding(X[:, :-1]) # (batch_size, src_len, embedding_size)

        batch_size, seq_len = X.shape

        hidden = self._bridge_state(hiddeEn, self.num_layers, self.hidden_size)
        cell   = self._bridge_state(cellEn, self.num_layers, self.hidden_size)

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

        return totalLoss / (seq_len - 1)

class LanguageModel(torch.nn.Module):
    def __init__(self, embed_size, hidden_size, word2indEng, word2indBg, startToken, unkToken, padToken, endToken,
                  lstm_layers, dropout_encoder, dropout_translator, dropaut_generator,
                  dropout_attention, maxlen=1000):
        super(LanguageModel, self).__init__()
        self.word2indBg = word2indBg
        self.word2indEng = word2indEng
        self.startTokenIdx = word2indBg[startToken]
        self.padTokenIdx = word2indBg[padToken]
        self.endTokenIdx = word2indBg[endToken]
        self.unkTokenIdx = word2indBg[unkToken]

        pos = torch.arange(maxlen)
        self.mask = pos.unsqueeze(0) > pos.unsqueeze(1)

        self.encoder = Encoder(len(word2indEng), embed_size, hidden_size, lstm_layers, dropout_encoder, self.padTokenIdx)
        self.transltor = DecoderTranslator(len(word2indBg), embed_size, hidden_size, lstm_layers, dropout_translator, dropout_attention, self.padTokenIdx)
        self.generator = DecoderGenerator(len(word2indEng), embed_size, hidden_size, lstm_layers, dropaut_generator, dropout_attention, self.padTokenIdx, self.mask)
        self.lossFn = nn.CrossEntropyLoss(ignore_index = self.padTokenIdx)
        self.dropout = nn.Dropout(dropout_encoder)

        # task weights (learnable parameters)
        device = next(self.parameters()).device
        self.task_weights = nn.Parameter(torch.ones(2).to(device))
        self.prev_losses = [torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)]

    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents_padded = [s+(m-len(s))*[self.padTokenIdx] for s in source]
        return torch.tensor(sents_padded, dtype=torch.long, device=device)	# shape=(batch_size, seq_len)

    def save(self,fileName):
        torch.save(self.state_dict(), fileName)

    def load(self,fileName):
        self.load_state_dict(torch.load(fileName))

    # def load(self,fileName, map_location):
    #     self.load_state_dict(torch.load(fileName, map_location))

    def forward(self, engBatch, bgBatch):
        engBatchPadded = self.preparePaddedBatch(engBatch)
        bgBatchPadded = self.preparePaddedBatch(bgBatch)

        engLength = [len(s) - 1 for s in engBatch]

        encoderOutputs, h, c = self.encoder(engBatchPadded, engLength)
        encoderOutputs.transpose_(0, 1)  # (seq_len, batch_size, 2 * hidden_size)

        encoderOutputs = self.dropout(encoderOutputs)

        # Current losses
        L1 = self.generator(engBatchPadded, encoderOutputs)  # Generation loss
        L2 = self.transltor(bgBatchPadded, encoderOutputs, h, c)  # Translation loss

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


    def isSourceSentenceComplete(self, sentence):
        return sentence.endswith(endToken)

    def generate(self, prefix, temperature=0.1, limit=1000):
        """
        Generate text continuation from a given prefix
        Args:
            prefix (list): List of tokens (words/indices) to start generation from
            limit (int): Maximum length of generated sequence
        Returns:
            list: Generated sequence of tokens
        """

        spEng = spm.SentencePieceProcessor(model_file=bpe_Eng)
        spBg = spm.SentencePieceProcessor(model_file=bpe_Bg)


        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            device = next(self.parameters()).device
            source = prefix
            prefix = [self.startTokenIdx] + spEng.encode(prefix, out_type=int) + [self.endTokenIdx]
            print(prefix)

            # if not self.isSourceSentenceComplete(source):
            #     prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=device)

            #     # Initialize generation with the prefix
            #     generated = prefix.copy()
            #     current_input = prefix_tensor

            #     hidden = torch.zeros(self.transltor.num_layers, 1, self.transltor.hidden_size).to(device)
            #     cell = torch.zeros(self.transltor.num_layers, 1, self.transltor.hidden_size).to(device)

            #     self.generator.eval()  # Set generator to evaluation mode

            #     encoder_outputs, hiddenEn, cellEn = self.encoder(prefix_tensor, [len(prefix) - 1])
            #     encoder_outputs.transpose_(0, 1)  # (seq_len, batch_size, hidden_size)

            #     output, (hidden, cell) = self.generator.rnn(
            #         self.generator.embedding(current_input[:, :-1]), (hidden, cell))

            #     # Generate tokens until END token or limit is reached
            #     for i in range(limit - len(prefix)):

            #         # Get embeddings
            #         emb = self.generator.embedding(current_input[:, -1:])  # Use last token as input

            #         # Run through RNN
            #         output, (hidden, cell) = self.generator.rnn(emb, (hidden, cell))

            #         # Apply attention
            #         hidden_attention = hidden[-1] if self.generator.num_layers > 1 else hidden.squeeze(0)
            #         att = self.generator.attention(hidden_attention, encoder_outputs)
            #         att = att.unsqueeze(1)
            #         weighted = torch.bmm(att, encoder_outputs)

            #         # Get prediction
            #         output = output.squeeze(1)
            #         weighted = weighted.squeeze(1)
            #         prediction = self.generator.fc(torch.cat((output, weighted), dim=1))
            #         prediction = torch.softmax(prediction / temperature, dim=-1)

            #         # Get next token
            #         next_token = torch.multinomial(prediction, 1).item()
            #         generated.append(next_token)

            #         # Stop if END token is generated
            #         if next_token == self.endTokenIdx:
            #             break

            #         # Update current input for next iteration
            #         current_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
            #         prefix_tensor = torch.cat((prefix_tensor, current_input), dim=1)
            #         encoderOutput, hiddenEn, cellEn = self.encoder.forwardStep(current_input, hiddenEn, cellEn)
            #         encoder_outputs = torch.cat((encoder_outputs, encoderOutput), dim=1)

            #     prefix = generated.copy()

            encoderOutputs, _, _ = self.encoder(
                torch.tensor([prefix], dtype=torch.long, device=device), [len(prefix) - 1])
            encoderOutputs = encoderOutputs.transpose(0, 1)

            hidden = torch.zeros(self.transltor.num_layers, 1, self.transltor.hidden_size).to(device)
            cell = torch.zeros(self.transltor.num_layers, 1, self.transltor.hidden_size).to(device)
            translated = []

            current_input = torch.tensor([[self.startTokenIdx]], dtype=torch.long, device=device)

            for t in range(limit - len(translated)):

                output, (hidden, cell) = self.transltor.rnn(self.transltor.embedding(current_input), (hidden, cell))
                # output: (1, 1, hidden_size)

                hidden_attention = hidden[-1] if self.transltor.num_layers > 1 else hidden.squeeze(0)  # (1, hidden_size)

                # Attention
                att = self.transltor.attention(hidden_attention, encoderOutputs)  # (1, seq_len)

                att = att.unsqueeze(1)  # (1, 1, seq_len)

                weighted = torch.bmm(att, encoderOutputs)  # (1, 1, hidden_size)

                # Prediction
                output = output.squeeze(1)  # (batch_size, hidden_size)

                weighted = weighted.squeeze(1)  # (batch_size, hidden_size)
                prediction = self.transltor.fc(torch.cat((output, weighted), dim=1)) # (batch_size, vocab_size)
                prediction = torch.softmax(prediction / temperature, dim=-1)

                current_input = torch.multinomial(prediction, 1)

                if current_input.item() == self.endTokenIdx:
                    break

                translated.append(current_input.item())

        resultGen = spEng.decode(prefix)
        resultTranslate = spBg.decode(translated)

        self.train()  # Set model back to training mode
        return resultGen, resultTranslate