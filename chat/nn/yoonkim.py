import torch
import torch.nn as nn
import torch.nn.functional as F


class Hyperparameters:
    kernel1 = 3
    kernel2 = 4
    kernel3 = 5
    n_filters = 300
    word_dim = 300
    vocab_size = 10_000
    max_sent_len = 4_330


class Conv1d(nn.Module):
    """ Conv1d => MaxPool1d

    Args:
        c_in: number of input channels
        c_out: number of output channels
        kernel_size: convolution kernel size
    """

    def __init__(self, c_in: int, c_out: int, kernel_size:int):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size)

    def forward(self, x):
        x = F.relu(self.conv(x))
        # Global max pool on all dims save batch.
        x = F.max_pool1d(x, x.size()[2:])
        return x


class YoonKimCNN(nn.Module):
    """ Yoon Kim's Text CNN

    Args:
        num_classes: number of classification classes
        hparams: hyperparameters of the model

    Reference:
        https://www.aclweb.org/anthology/D14-1181
    """

    def __init__(self, num_classes, hparams, frozen=False):
        super(YoonKimCNN, self).__init__()
        self.hp = hparams

        self.embedding = nn.Embedding(hparams.vocab_size, hparams.word_dim, padding_idx=0)
        self.conv1 = Conv1d(hparams.word_dim, hparams.n_filters, hparams.kernel1)
        self.conv2 = Conv1d(hparams.word_dim, hparams.n_filters, hparams.kernel2)
        self.conv3 = Conv1d(hparams.word_dim, hparams.n_filters, hparams.kernel3)
        self.fc = nn.Linear(self._sum_filters(), num_classes)

        self._initialize_params()

    def load_embeddings(self, embeddings, frozen=False):
        """ Load pretrained embeddings """
        self.embedding.load_state_dict({'weight': embeddings})

        if frozen:
            self.embedding.weight.requires_grad = False

    def _initialize_params(self):
        for name, param in self.named_parameters():
            try:
                nn.init.xavier_uniform_(param)
            except:
                nn.init.constant_(param, 0)

    def _sum_filters(self):
        """Total number of convolution filters for the three layers"""
        return self.hp.n_filters * 3

    def loss_value(self, y_pred, y_true):
        """ Calculate a value of loss function """
        return F.binary_cross_entropy_with_logits(y_pred, y_true)

    def forward(self, x):
        x = self.embedding(x)
        # Make sure self.hp.word_dim is the channel dim for convolution
        x = x.transpose(1, 2)

        conv_results = []
        conv_results.append(self.conv1(x).view(-1, self.hp.n_filters))
        conv_results.append(self.conv2(x).view(-1, self.hp.n_filters))
        conv_results.append(self.conv2(x).view(-1, self.hp.n_filters))
        x = torch.cat(conv_results, 1)
        logits = self.fc(x)
        return logits

