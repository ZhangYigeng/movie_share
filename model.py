import torch.nn as nn


class LSTMmodel(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        # self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        lstm_out, hidden = self.lstm(x, hidden)

        # print(hidden)
        # print(hidden[0].size())
        print(lstm_out.size())

        # choose out/hidden as output
        # out = lstm_out
        out = hidden[0].permute(1, 0, 2)
        out = out[:, -1, :].view(batch_size, -1)

        # stack up lstm outputs
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(out)
        out = self.fc(out)
        # sigmoid function
        # sig_out = self.sig(out)
        # sig_out = sig_out[:, -1,:]
        # out = out[:, -1,:]

        # reshape to be batch_size first
        # sig_out = sig_out.view(batch_size, -1)
        # sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        # return sig_out, hidden

        return out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new_zeros(self.n_layers, batch_size, self.hidden_dim),
                  weight.new_zeros(self.n_layers, batch_size, self.hidden_dim))

        return hidden
