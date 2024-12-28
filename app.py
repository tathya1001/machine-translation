from flask import Flask, render_template, request, jsonify
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from flask_cors import CORS  
import urllib.parse

app = Flask(__name__)

CORS(app)

vocab_de = torch.load('vocab_de.pth')
vocab_en = torch.load('vocab_en.pth')

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, dropout_p=0.1):
        super().__init__()
        self.e = nn.Embedding(input_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x, lengths):
        x = self.e(x)
        x = self.dropout(x)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.gru(x)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size):
        super().__init__()
        self.e = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout()
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, batch_first=True)
        self.lin = nn.Linear(hidden_size, output_size)
        self.lsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, context, prev_hidden):
        x = self.e(x)
        x = self.dropout(x)
        x = torch.cat((x, context), dim=2)
        output, hidden = self.gru(x, prev_hidden)
        y = self.lin(output)
        y = self.lsoftmax(y)
        return y, hidden

class Bahdanau_Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, new_hidden_size):
        super().__init__()
        self.eh2nh = nn.Linear(in_features=encoder_hidden_size, out_features=new_hidden_size)
        self.dh2nh = nn.Linear(in_features=decoder_hidden_size, out_features=new_hidden_size)
        self.score = nn.Linear(in_features=new_hidden_size, out_features=1)

    def forward(self, query, keys):
        query = self.dh2nh(query)
        keys = self.eh2nh(keys)
        att_score = self.score(torch.tanh(query.permute(1, 0, 2) + keys))
        att_score = att_score.squeeze(2).unsqueeze(1)
        att_weights = F.softmax(att_score, dim=-1)
        context = torch.bmm(att_weights, keys)
        return context, att_weights

encoder = Encoder(input_size=len(vocab_de), embed_size=300, hidden_size=512)
decoder = Decoder(output_size=len(vocab_en), embed_size=300, hidden_size=512)
ba = Bahdanau_Attention(encoder_hidden_size=512, decoder_hidden_size=512, new_hidden_size=512)

encoder.load_state_dict(torch.load("encoder.pth", map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load("decoder.pth", map_location=torch.device('cpu')))
ba.load_state_dict(torch.load("ba.pth", map_location=torch.device('cpu')))

UNK_IDX = 0
PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

@app.route('/<sentence>', methods=['GET'])
def get_token_ids(sentence):
    decoded_sentence = urllib.parse.unquote(sentence)
    tokens = decoded_sentence.split()
    input_ids = [vocab_de.get_stoi().get(word, UNK_IDX) for word in tokens]
    input_tensor = torch.tensor(input_ids).unsqueeze(0)
    lengths = torch.tensor([len(input_ids)])
    
    encoder_outputs, encoder_hidden = encoder(input_tensor, lengths)
    
    decoder_input = torch.tensor([[BOS_IDX]])
    decoder_hidden = encoder_hidden
    translated_tokens = []
    context = torch.zeros((1, 1, 512))
    
    for _ in range(50):
        attention_context, _ = ba(decoder_hidden.permute(1, 0, 2), encoder_outputs)
        decoder_output, decoder_hidden = decoder(decoder_input, attention_context, decoder_hidden)
        next_token = torch.argmax(decoder_output, dim=2)
        next_token_id = next_token.item()
        
        if next_token_id == EOS_IDX:
            break
        
        translated_tokens.append(next_token_id)
        decoder_input = next_token
        
    translated_words = [vocab_en.get_itos()[token] for token in translated_tokens]
    translated_sentence = " ".join(translated_words)
    return jsonify({'translated_sentence': translated_sentence})

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
